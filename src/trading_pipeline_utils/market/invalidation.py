from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from trading_pipeline_utils.market.curve import TranslatedProduct
from trading_pipeline_utils.market.signals import Signal
from trading_pipeline_utils.settings import InvalidationConfig, PostModelConfig

logger = logging.getLogger(__name__)


@dataclass
class InvalidationResult:
    product_code: str
    invalidation_flag: bool
    invalidation_reasons: list[str] = field(default_factory=list)
    hold_or_exit: str = "hold"
    severity: float = 0.0


def _check_fair_value_shift(
    signal: Signal,
    product: TranslatedProduct,
    prior_fair_value: float | None,
    cfg: InvalidationConfig,
) -> str | None:
    if prior_fair_value is None:
        return None
    shift = abs(product.fair_value - prior_fair_value)
    threshold = cfg.sigma_shift_threshold * max(product.std, 1e-6)
    if shift > threshold:
        return (
            f"Fair value shifted by {shift:.2f} EUR/MWh "
            f"(>{cfg.sigma_shift_threshold:.1f} sigma = {threshold:.2f})"
        )
    return None


def _check_edge_disappeared(
    signal: Signal,
    product: TranslatedProduct,
    cfg: InvalidationConfig,
) -> str | None:
    if not product.has_market_forward:
        return None
    if product.p10 is not None and product.p90 is not None:
        if product.p10 <= product.market_forward <= product.p90:
            if abs(signal.edge_z) < 0.5:
                return "Market forward is inside the fair-value band — edge has disappeared"
    return None


def _check_beta_stability(
    product: TranslatedProduct,
    cfg: InvalidationConfig,
) -> str | None:
    diag = product.diagnostics
    if diag.translation_quality == "direct":
        return None
    if diag.beta_stability < cfg.beta_stability_min:
        return (
            f"Beta stability = {diag.beta_stability:.2f} "
            f"(below {cfg.beta_stability_min:.2f} threshold)"
        )
    return None


def _check_r2(
    product: TranslatedProduct,
    cfg: InvalidationConfig,
) -> str | None:
    diag = product.diagnostics
    if diag.translation_quality == "direct":
        return None
    if diag.rolling_r2 < cfg.r2_min:
        return (
            f"Translation R² = {diag.rolling_r2:.3f} "
            f"(below {cfg.r2_min:.3f} threshold)"
        )
    return None


def _check_uncertainty_width(
    product: TranslatedProduct,
    cfg: InvalidationConfig,
) -> str | None:
    if product.p10 is not None and product.p90 is not None:
        width = product.p90 - product.p10
        if product.std > 0 and width > cfg.max_uncertainty_multiple * product.std * 2:
            return f"Uncertainty band is unusually wide: p10–p90 = {width:.2f} EUR/MWh"
    return None


def _check_model_skill(
    signal: Signal,
    cfg: InvalidationConfig,
) -> str | None:
    if signal.model_skill_weight < cfg.skill_degradation_threshold:
        return (
            f"Model skill weight = {signal.model_skill_weight:.2f} "
            f"(below {cfg.skill_degradation_threshold:.2f})"
        )
    return None


def _check_coverage(
    product: TranslatedProduct,
    cfg: InvalidationConfig,
) -> str | None:
    if product.diagnostics.coverage_ratio < cfg.min_coverage_ratio:
        return (
            f"Coverage ratio = {product.diagnostics.coverage_ratio:.2f} "
            f"(below {cfg.min_coverage_ratio:.2f})"
        )
    return None


def evaluate_invalidation(
    signal: Signal,
    product: TranslatedProduct,
    config: PostModelConfig,
    *,
    prior_fair_value: float | None = None,
    signal_age_hours: float = 0.0,
) -> InvalidationResult:
    cfg = config.invalidation
    reasons: list[str] = []

    for check in [
        _check_fair_value_shift(signal, product, prior_fair_value, cfg),
        _check_edge_disappeared(signal, product, cfg),
        _check_beta_stability(product, cfg),
        _check_r2(product, cfg),
        _check_uncertainty_width(product, cfg),
        _check_model_skill(signal, cfg),
        _check_coverage(product, cfg),
    ]:
        if check is not None:
            reasons.append(check)

    if signal_age_hours > cfg.max_signal_age_hours:
        reasons.append(
            f"Signal age = {signal_age_hours:.0f}h (exceeds {cfg.max_signal_age_hours}h max)"
        )

    flag = len(reasons) > 0
    severity = float(np.clip(len(reasons) / 5.0, 0.0, 1.0))
    hold_or_exit = "exit" if len(reasons) >= 3 else ("reduce" if flag else "hold")

    return InvalidationResult(
        product_code=product.product_code,
        invalidation_flag=flag,
        invalidation_reasons=reasons,
        hold_or_exit=hold_or_exit,
        severity=severity,
    )


def evaluate_all_invalidations(
    signals: dict[str, Signal],
    products: dict[str, TranslatedProduct],
    config: PostModelConfig,
    *,
    prior_fair_values: dict[str, float] | None = None,
) -> dict[str, InvalidationResult]:
    priors = prior_fair_values or {}
    return {
        code: evaluate_invalidation(
            sig,
            products[code],
            config,
            prior_fair_value=priors.get(code),
        )
        for code, sig in signals.items()
        if code in products
    }
