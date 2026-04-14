"""Rolling-origin daily backtest with train/val gap and holdout reporting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from trading_pipeline_utils.features.engineering import build_feature_matrix
from trading_pipeline_utils.features.transforms import ArcsinhPriceTransform
from trading_pipeline_utils.forecasting.metrics import enforce_quantile_monotonicity, metrics_block
from trading_pipeline_utils.models.training import compute_sample_weights, train_lightgbm_suite
from trading_pipeline_utils.settings import ModelPipelineConfig

logger = logging.getLogger(__name__)


def iter_origin_folds(
    n_rows: int,
    config: ModelPipelineConfig,
    *,
    min_train_rows: int = 400,
) -> Iterator[tuple[slice, slice]]:
    bc = config.backtest
    gap = bc.gap_hours
    vb = bc.validation_block_hours
    step = bc.origin_step_hours
    hold = bc.holdout_tail_hours or 0
    effective_n = n_rows - hold
    o = min_train_rows
    while True:
        v0 = o + gap
        v1 = v0 + vb
        if v1 > effective_n:
            break
        if bc.window_type == "expanding":
            t0 = 0
        else:
            roll = bc.rolling_train_hours or 24 * 365
            t0 = max(0, o - roll)
        yield slice(t0, o), slice(v0, v1)
        o += step


def run_rolling_backtest(
    df: pd.DataFrame,
    config: ModelPipelineConfig,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    c = config.columns
    n = len(df)
    oof_rows: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []

    for fi, (tr_sl, va_sl) in enumerate(iter_origin_folds(n, config)):
        tr_idx = np.zeros(n, dtype=bool)
        va_idx = np.zeros(n, dtype=bool)
        tr_idx[tr_sl] = True
        va_idx[va_sl] = True

        price_tr = df.loc[tr_idx, c.price_da].dropna()
        if len(price_tr) < 50:
            continue
        p95 = float(np.percentile(np.abs(price_tr.values), 95))

        X, _num_cols, cat_cols = build_feature_matrix(df, config, train_abs_price_p95=p95)
        price_vec = df[c.price_da].astype(np.float64).values
        row_ok = ~X.isna().all(axis=1).values

        tr_row = tr_idx & row_ok & np.isfinite(price_vec)
        va_row = va_idx & row_ok & np.isfinite(price_vec)
        if int(va_row.sum()) < 24:
            continue

        tfm = ArcsinhPriceTransform(scale_floor=config.target.scale_floor)
        tfm.fit(pd.Series(price_vec[tr_row]))
        y_t = np.full(n, np.nan, dtype=np.float64)
        y_t[tr_row] = tfm.transform(price_vec[tr_row])

        sw = compute_sample_weights(
            df[c.price_da],
            tr_idx,
            df.index,
            neg_price_extra=config.sample_weight.neg_price_extra,
            half_life_days=config.sample_weight.half_life_days,
        )

        tr_pos = np.flatnonzero(tr_row)
        if len(tr_pos) < 120:
            continue
        split_at = max(len(tr_pos) - 28 * 24, int(len(tr_pos) * 0.88))
        inner_fit = np.zeros(n, dtype=bool)
        inner_es = np.zeros(n, dtype=bool)
        inner_fit[tr_pos[:split_at]] = True
        inner_es[tr_pos[split_at:]] = True

        fit_ix = np.flatnonzero(inner_fit)
        es_ix = np.flatnonzero(inner_es)
        bundle = train_lightgbm_suite(
            X.iloc[fit_ix],
            y_t[fit_ix],
            X.iloc[es_ix],
            y_t[es_ix],
            sw[inner_fit],
            cat_cols,
            config,
        )

        va_ix = np.flatnonzero(va_row)
        va_x = X.iloc[va_ix]
        y_va = price_vec[va_ix]
        pred_t = bundle.lgbm_point.predict(va_x)
        q10_t = bundle.lgbm_q10.predict(va_x)
        q50_t = bundle.lgbm_q50.predict(va_x)
        q90_t = bundle.lgbm_q90.predict(va_x)

        pt = tfm.inverse_transform(pred_t)
        p10 = tfm.inverse_transform(q10_t)
        p50 = tfm.inverse_transform(q50_t)
        p90 = tfm.inverse_transform(q90_t)
        p10, p50, p90 = enforce_quantile_monotonicity(p10, p50, p90)

        idx = df.index[va_ix]
        block = pd.DataFrame(
            {
                "fold": fi,
                "y_actual": y_va,
                "y_point": pt,
                "q10": p10,
                "q50": p50,
                "q90": p90,
            },
            index=idx,
        )
        oof_rows.append(block)

        sw_va = sw[va_ix]
        mb = metrics_block(
            y_va,
            pt,
            sw_va,
            p10,
            p50,
            p90,
            index=idx,
            delivery_timezone=config.validation.delivery_timezone,
            peak_hour_start=config.simulation.peak_local_hour_start,
            peak_hour_end=config.simulation.peak_local_hour_end,
        )
        mb["fold"] = fi
        fold_metrics.append(mb)

    oof = pd.concat(oof_rows).sort_index()
    oof = oof[~oof.index.duplicated(keep="last")]

    fold_df = pd.DataFrame(fold_metrics)
    out: dict[str, Any] = {"oof_predictions": oof, "fold_metrics": fold_df}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(output_dir / config.outputs.fold_metrics_csv, index=False)
        oof.to_parquet(output_dir / config.outputs.oof_predictions_parquet)

    return out
