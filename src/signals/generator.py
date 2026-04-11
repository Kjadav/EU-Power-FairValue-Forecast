from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Signal:
    direction: str
    outlook: str
    confidence: float
    summary: str


def generate(
    baseline_forecast: np.ndarray | list[float],
    model_forecast: np.ndarray | list[float],
    model_metrics: dict[str, Any],
    config: dict[str, Any],
) -> Signal:
    """Compare model forecast to baseline; derive directional signal and outlook."""
    baseline = np.asarray(baseline_forecast, dtype=np.float64)
    model = np.asarray(model_forecast, dtype=np.float64)

    baseline_mean = float(np.mean(baseline))
    model_mean = float(np.mean(model))
    spread = model_mean - baseline_mean
    pct_diff = spread / baseline_mean if baseline_mean != 0 else 0.0

    threshold = config.get("model", {}).get("signal_threshold_pct", 0.02)

    if pct_diff > threshold:
        direction, outlook = "LONG", "Bullish"
    elif pct_diff < -threshold:
        direction, outlook = "SHORT", "Bearish"
    else:
        direction, outlook = "NEUTRAL", "Uncertain"

    mae = model_metrics.get("mae", 0.0)
    confidence = max(0.0, min(1.0, 1.0 - (mae / baseline_mean if baseline_mean else 1.0)))

    rmse = model_metrics.get("rmse", 0.0)
    summary = (
        f"Model mean={model_mean:.2f}, baseline mean={baseline_mean:.2f}, "
        f"spread={spread:+.2f} ({pct_diff:+.1%}), MAE={mae:.2f}, RMSE={rmse:.2f}"
    )
    return Signal(
        direction=direction,
        outlook=outlook,
        confidence=round(confidence, 3),
        summary=summary,
    )
