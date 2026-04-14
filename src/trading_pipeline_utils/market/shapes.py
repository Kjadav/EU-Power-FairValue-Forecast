from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from trading_pipeline_utils.market.curve import TranslatedProduct
from trading_pipeline_utils.settings import PostModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ShapeView:
    shape_name: str
    long_leg: str
    short_leg: str
    model_shape: float
    market_shape: float
    shape_edge: float
    shape_zscore: float
    signal_direction: str
    suggested_expression: str


def _zscore(edge: float, std: float, eps: float = 1e-6) -> float:
    return edge / max(std, eps)


def _direction(edge: float, std: float, threshold: float, eps: float = 1e-6) -> str:
    if std < eps:
        return "flat"
    if edge > threshold * std:
        return "long"
    if edge < -threshold * std:
        return "short"
    return "flat"


def compute_shape_views(
    products: dict[str, TranslatedProduct],
    config: PostModelConfig,
) -> list[ShapeView]:
    eps = config.signals.epsilon
    threshold = config.signals.entry_threshold
    views: list[ShapeView] = []

    shape_pairs = [
        ("peak_vs_base_week", "prompt_week_peak", "prompt_week_base"),
        ("week_vs_month_base", "prompt_week_base", "prompt_month_base"),
        ("month_vs_quarter_base", "prompt_month_base", "prompt_quarter_base"),
        ("week_vs_month_peak", "prompt_week_peak", "prompt_month_peak"),
        ("month_vs_quarter_peak", "prompt_month_peak", "prompt_quarter_peak"),
    ]

    for name, long_code, short_code in shape_pairs:
        long_prod = products.get(long_code)
        short_prod = products.get(short_code)
        if long_prod is None or short_prod is None:
            continue

        model_shape = long_prod.fair_value - short_prod.fair_value
        market_shape = long_prod.market_forward - short_prod.market_forward
        shape_edge = model_shape - market_shape

        combined_std = float(np.sqrt(long_prod.std ** 2 + short_prod.std ** 2))
        z = _zscore(shape_edge, combined_std, eps)
        dir_ = _direction(shape_edge, combined_std, threshold, eps)

        if dir_ == "long":
            expr = f"Long {long_code} / Short {short_code}: shape edge = {shape_edge:+.2f} EUR/MWh (z={z:+.2f})."
        elif dir_ == "short":
            expr = f"Short {long_code} / Long {short_code}: shape edge = {shape_edge:+.2f} EUR/MWh (z={z:+.2f})."
        else:
            expr = f"Flat {name}: shape edge is small ({shape_edge:+.2f}, z={z:+.2f})."

        views.append(ShapeView(
            shape_name=name,
            long_leg=long_code,
            short_leg=short_code,
            model_shape=model_shape,
            market_shape=market_shape,
            shape_edge=shape_edge,
            shape_zscore=z,
            signal_direction=dir_,
            suggested_expression=expr,
        ))

    return views
