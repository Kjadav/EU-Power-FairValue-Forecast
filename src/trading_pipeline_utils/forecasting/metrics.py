from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def enforce_quantile_monotonicity(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(q10, dtype=np.float64)
    b = np.asarray(q50, dtype=np.float64)
    c = np.asarray(q90, dtype=np.float64)
    b = np.maximum(b, a)
    c = np.maximum(c, b)
    return a, b, c


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    e = y - q_pred
    return float(np.mean(np.maximum(alpha * e, (alpha - 1.0) * e)))


def weekly_baseload_peakload(prices_168: np.ndarray, peak_mask: np.ndarray) -> tuple[float, float]:
    base = float(np.mean(prices_168))
    pm = np.asarray(peak_mask, dtype=bool)
    peak_ld = float(np.mean(prices_168[pm])) if pm.any() else float("nan")
    return base, peak_ld


def metrics_block(
    y: np.ndarray,
    y_point: np.ndarray,
    sample_weight: np.ndarray | None,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    *,
    index: pd.DatetimeIndex | None = None,
    delivery_timezone: str = "Europe/Berlin",
    peak_hour_start: int = 8,
    peak_hour_end: int = 20,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    y_point = np.asarray(y_point, dtype=np.float64)
    p10 = np.asarray(p10, dtype=np.float64)
    p50 = np.asarray(p50, dtype=np.float64)
    p90 = np.asarray(p90, dtype=np.float64)
    err = y - y_point
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    med_ae = float(np.median(np.abs(err)))

    wmae = mae
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64)
        sw = np.sum(w)
        if sw > 0:
            wmae = float(np.sum(w * np.abs(err)) / sw)

    pin_10 = pinball_loss(y, p10, 0.1)
    pin_50 = pinball_loss(y, p50, 0.5)
    pin_90 = pinball_loss(y, p90, 0.9)

    cover = float(np.mean((y >= p10) & (y <= p90)))

    neg_m = y < 0
    mae_neg = float(np.mean(np.abs(err[neg_m]))) if neg_m.any() else float("nan")

    ad = np.abs(y)
    thr = np.percentile(ad, 90) if len(ad) else 0.0
    hi = ad >= thr
    mae_hi = float(np.mean(np.abs(err[hi]))) if hi.any() else float("nan")

    weekly_base_err = float("nan")
    weekly_peak_err = float("nan")
    if index is not None and len(index) == len(y):
        loc = index.tz_convert(delivery_timezone)
        peak_m = np.array(
            [
                (d.dayofweek < 5) and (peak_hour_start <= d.hour < peak_hour_end)
                for d in loc
            ],
            dtype=bool,
        )
        w = 168
        nw = len(y) // w
        be: list[float] = []
        pe: list[float] = []
        for i in range(nw):
            sl = slice(i * w, (i + 1) * w)
            ya = y[sl]
            yp = y_point[sl]
            pm = peak_m[sl]
            ba, _ = weekly_baseload_peakload(ya, pm)
            bp, _ = weekly_baseload_peakload(yp, pm)
            be.append(abs(ba - bp))
            _, pa = weekly_baseload_peakload(ya, pm)
            _, pp = weekly_baseload_peakload(yp, pm)
            if np.isfinite(pa) and np.isfinite(pp):
                pe.append(abs(pa - pp))
        weekly_base_err = float(np.mean(be)) if be else float("nan")
        weekly_peak_err = float(np.mean(pe)) if pe else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "wmae": wmae,
        "median_ae": med_ae,
        "pinball_q10": pin_10,
        "pinball_q50": pin_50,
        "pinball_q90": pin_90,
        "coverage_p10_p90": cover,
        "mae_negative_hours": mae_neg,
        "mae_top_abs_decile": mae_hi,
        "weekly_baseload_mae": weekly_base_err,
        "weekly_peakload_mae": weekly_peak_err,
    }


def backtest_report(actual: pd.Series, forecast: pd.Series) -> dict[str, Any]:
    """mae/ rmse is returned"""
    aligned = pd.concat([actual, forecast], axis=1, keys=["y", "y_hat"]).dropna()
    err = aligned["y"] - aligned["y_hat"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {"n": len(aligned), "mae": mae, "rmse": rmse}
