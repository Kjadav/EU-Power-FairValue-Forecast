from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from trading_pipeline_utils.market.curve import TranslatedProduct
from trading_pipeline_utils.settings import PostModelConfig, SignalConfig

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    product_code: str
    edge: float
    edge_z: float
    risk_premium_proxy: float
    confidence: float
    signal_score: float
    direction: str  # "long" | "short" | "flat"
    suggested_position_units: float
    suggested_expression: str
    model_skill_weight: float
    translation_stability_weight: float
    coverage_weight: float
    distribution_weight: float


def _model_skill_weight(backtest_summary: dict[str, Any]) -> float:
    """Derive [0,1] skill weight from backtest metrics.

    Heuristic: scale MAE into a skill score.  Lower MAE → higher skill.
    Calibrated to typical DE DA price range (20-200 EUR/MWh).
    """
    mae = backtest_summary.get("overall_mae_eur_mwh")
    if mae is None or not np.isfinite(mae):
        return 0.5
    # MAE=0 → 1.0, MAE=30 → 0.5, MAE=60 → 0.25
    return float(np.clip(1.0 / (1.0 + mae / 30.0), 0.0, 1.0))


def _translation_stability_weight(product: TranslatedProduct) -> float:
    diag = product.diagnostics
    if diag.translation_quality == "direct":
        return 1.0
    if diag.translation_quality == "good":
        return 0.9
    if diag.translation_quality == "fair":
        return 0.7
    if diag.translation_quality == "weak":
        return 0.4
    if diag.translation_quality == "fallback":
        return 0.3
    return 0.5


def _coverage_weight(product: TranslatedProduct) -> float:
    c = product.diagnostics.coverage_ratio
    if c <= 0 or not np.isfinite(c):
        return 0.5
    return float(np.clip(c, 0.0, 1.0))


def _distribution_weight(product: TranslatedProduct) -> float:
    """Higher weight when market price sits in the tails of the distribution."""
    if product.p10 is None or product.p90 is None:
        return 0.5
    spread = product.p90 - product.p10
    if spread <= 0:
        return 0.5
    mkt = product.market_forward
    if mkt < product.p10:
        return 1.0
    if mkt > product.p90:
        return 1.0
    dist_from_center = abs(mkt - (product.p10 + product.p90) / 2)
    return float(np.clip(0.5 + dist_from_center / spread, 0.5, 1.0))


def _direction(edge: float, std: float, threshold: float, eps: float) -> str:
    if std < eps:
        return "flat" if abs(edge) < eps else ("long" if edge > 0 else "short")
    if edge > threshold * std:
        return "long"
    if edge < -threshold * std:
        return "short"
    return "flat"


def _expression_text(product_code: str, direction: str, edge: float, edge_z: float) -> str:
    if direction == "flat":
        return f"Flat {product_code}: edge is small (z={edge_z:+.2f})."
    verb = "Long" if direction == "long" else "Short"
    return (
        f"{verb} {product_code}: translated fair value is "
        f"{abs(edge_z):.1f} sigma {'above' if edge > 0 else 'below'} market."
    )


def compute_signal(
    product: TranslatedProduct,
    backtest_summary: dict[str, Any],
    config: PostModelConfig,
) -> Signal:
    sc = config.signals
    eps = sc.epsilon

    std = max(product.std, eps)
    edge = product.edge
    edge_z = edge / std

    msw = _model_skill_weight(backtest_summary)
    tsw = _translation_stability_weight(product)
    cw = _coverage_weight(product)
    dw = _distribution_weight(product)

    raw_confidence = float(np.clip(abs(edge_z) / sc.z_cap, 0.0, 1.0))
    confidence = raw_confidence * msw * tsw * cw * dw
    confidence = float(np.clip(confidence, 0.0, 1.0))

    signal_score = float(np.sign(edge)) * confidence
    dir_ = _direction(edge, std, sc.entry_threshold, eps)

    vol = std
    pos = float(np.clip(
        sc.risk_budget * signal_score / max(vol, eps),
        -sc.max_units,
        sc.max_units,
    ))

    return Signal(
        product_code=product.product_code,
        edge=edge,
        edge_z=edge_z,
        risk_premium_proxy=product.risk_premium_proxy,
        confidence=confidence,
        signal_score=signal_score,
        direction=dir_,
        suggested_position_units=pos,
        suggested_expression=_expression_text(product.product_code, dir_, edge, edge_z),
        model_skill_weight=msw,
        translation_stability_weight=tsw,
        coverage_weight=cw,
        distribution_weight=dw,
    )


def compute_all_signals(
    products: dict[str, TranslatedProduct],
    backtest_summary: dict[str, Any],
    config: PostModelConfig,
) -> dict[str, Signal]:
    return {
        code: compute_signal(prod, backtest_summary, config)
        for code, prod in products.items()
    }
