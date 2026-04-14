from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from trading_pipeline_utils.features.transforms import ArcsinhPriceTransform
from trading_pipeline_utils.forecasting.metrics import weekly_baseload_peakload
from trading_pipeline_utils.settings import ModelPipelineConfig

logger = logging.getLogger(__name__)

Z_SPAN_NORMAL = 2.563103


def compute_standardized_residuals(
    y_actual: np.ndarray,
    y_point_t: np.ndarray,
    q10_t: np.ndarray,
    q90_t: np.ndarray,
    tfm: ArcsinhPriceTransform,
    sigma_floor: float,
) -> np.ndarray:
    ya = tfm.transform(y_actual)
    sigma = np.maximum((q90_t - q10_t) / Z_SPAN_NORMAL, sigma_floor)
    return (ya - y_point_t) / np.maximum(sigma, 1e-9)


def encode_bucket(hour_local: np.ndarray, weekend: np.ndarray, month: np.ndarray) -> np.ndarray:
    return (hour_local + 24 * weekend + 100 * month).astype(np.int64)


def simulate_paths_next_week(
    yhat_t: np.ndarray,
    q10_t: np.ndarray,
    q90_t: np.ndarray,
    tfm: ArcsinhPriceTransform,
    z_train: np.ndarray,
    bucket_ids: np.ndarray,
    train_bucket_ids: np.ndarray,
    config: ModelPipelineConfig,
    *,
    future_index: pd.DatetimeIndex,
    peak_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    sc = config.simulation
    if sc.simulation_mode == "single_origin_week" and not sc.perfect_foresight_mode:
        raise ValueError(
            "single_origin_week requires perfect_foresight_mode=True unless multi-vintage "
            "ex-ante fundamentals are supplied (not implemented). "
            "Use simulation_mode='d_minus_1_recursive' for recursive D-1 style paths."
        )

    H = len(yhat_t)
    rng = np.random.default_rng(config.lgbm.random_state)
    sigma = np.maximum((q90_t - q10_t) / Z_SPAN_NORMAL, sc.sigma_floor)

    pools: dict[int, np.ndarray] = {}
    for b in np.unique(train_bucket_ids):
        pools[int(b)] = z_train[train_bucket_ids == b]
    global_pool = z_train[np.isfinite(z_train)]

    z_clip = getattr(sc, "z_clip", 3.5)
    price_floor = getattr(sc, "price_floor_eur", -500.0)
    price_ceil = getattr(sc, "price_ceil_eur", 1000.0)

    paths = np.zeros((sc.n_paths, H), dtype=np.float64)
    for p in range(sc.n_paths):
        for h in range(H):
            yh = float(yhat_t[h])
            sh = float(sigma[h])
            if sc.deterministic:
                dev = 0.0
            elif sc.use_gaussian_shocks:
                z_draw = float(rng.standard_normal())
                z_draw = float(np.clip(z_draw, -z_clip, z_clip))
                dev = z_draw * sh
            else:
                bid = int(bucket_ids[h])
                pool = pools.get(bid, global_pool)
                if len(pool) == 0:
                    pool = global_pool
                z_draw = float(rng.choice(pool))
                z_draw = float(np.clip(z_draw, -z_clip, z_clip))
                dev = z_draw * sh
            price = float(tfm.inverse_transform(np.array([yh + dev]))[0])
            paths[p, h] = float(np.clip(price, price_floor, price_ceil))

    if peak_mask is None:
        loc = future_index.tz_convert(config.validation.delivery_timezone)
        peak_m = np.array(
            [
                (d.dayofweek < 5)
                and (sc.peak_local_hour_start <= d.hour < sc.peak_local_hour_end)
                for d in loc
            ],
            dtype=bool,
        )
    else:
        peak_m = np.asarray(peak_mask, dtype=bool)
        if len(peak_m) != H:
            raise ValueError("peak_mask length must match horizon")

    # Index 0: arithmetic mean of all H hourly DA prices on the path (= weekly average DA for that path).
    base_w = np.array([weekly_baseload_peakload(paths[i], peak_m)[0] for i in range(sc.n_paths)])
    peak_w = np.array([weekly_baseload_peakload(paths[i], peak_m)[1] for i in range(sc.n_paths)])

    out: dict[str, Any] = {
        "paths": paths,
        "hourly_mean": np.mean(paths, axis=0),
        "hourly_std": np.std(paths, axis=0),
        "hourly_p05": np.percentile(paths, 5, axis=0),
        "hourly_p50": np.percentile(paths, 50, axis=0),
        "hourly_p95": np.percentile(paths, 95, axis=0),
        "weekly_baseload_mean": float(np.mean(base_w)),
        "weekly_baseload_std": float(np.std(base_w)),
        "weekly_baseload_p05": float(np.percentile(base_w, 5)),
        "weekly_baseload_p50": float(np.percentile(base_w, 50)),
        "weekly_baseload_p95": float(np.percentile(base_w, 95)),
        "weekly_peakload_mean": float(np.nanmean(peak_w)),
        "weekly_peakload_std": float(np.nanstd(peak_w)),
        "weekly_peakload_p05": float(np.nanpercentile(peak_w, 5)),
        "weekly_peakload_p50": float(np.nanpercentile(peak_w, 50)),
        "weekly_peakload_p95": float(np.nanpercentile(peak_w, 95)),
    }
    ref = sc.reference_market_price
    if ref is not None:
        out["prob_baseload_gt_ref"] = float(np.mean(base_w > ref))
        out["prob_peakload_gt_ref"] = float(np.mean(peak_w > ref))
    return out
