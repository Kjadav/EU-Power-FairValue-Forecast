"""Walk-forward: train OLS on all hours before day D; forecast D’s 24 hourly DA prices."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from eu_power_forecast.forecasting.naive_baseline_model import naive_price_forecast
from eu_power_forecast.forecasting.validation import backtest_report
from eu_power_forecast.forecasting.regression import (
    PRICE_COL,
    add_autoregressive_price_lags,
    build_design_xy,
    dates_with_full_24h,
    fit_ols_beta,
    load_day_ahead_regression_panel,
    predict_ols,
    row_mask_valid_xy,
)


def _utc_midnight(d: date) -> pd.Timestamp:
    return pd.Timestamp(datetime(d.year, d.month, d.day, tzinfo=timezone.utc))


def walk_forward_day_ahead_backtest(
    bundle_dir: str | Path,
    *,
    n_eval_days: int = 7,
    min_train_hours: int = 200,
) -> dict[str, Any]:
    """
    For each evaluation day (last ``n_eval_days`` UTC days with 24 hourly rows):

    * Train OLS on every valid row with timestamp strictly before that day.
    * Predict the 24 hours of that day (tomorrow-style block in backtest).
    * Compare to naive last hour and naive seasonal (price shifted by 24h).

    No leakage: training excludes the forecast day; lags only use past prices.
    """
    bundle_dir = Path(bundle_dir)
    panel = add_autoregressive_price_lags(load_day_ahead_regression_panel(bundle_dir))
    X, y = build_design_xy(panel)
    valid = row_mask_valid_xy(X, y)
    if not valid.any():
        raise ValueError(
            "no usable regression rows: P_{t-168} needs >168 hours of aligned hourly data "
            "(price, residual load, hydro forecast). Re-fetch a longer SMARD slice."
        )

    complete = dates_with_full_24h(panel.index)
    if not complete:
        raise ValueError("no UTC calendar day with 24 hourly observations")

    eval_dates = complete[-n_eval_days:] if len(complete) >= n_eval_days else complete

    reg_chunks: list[pd.Series] = []
    naive_last_chunks: list[pd.Series] = []
    naive_seas_chunks: list[pd.Series] = []
    actual_chunks: list[pd.Series] = []
    per_day: list[dict[str, Any]] = []

    for eval_d in eval_dates:
        day0 = _utc_midnight(eval_d)
        day1 = day0 + pd.Timedelta(days=1)

        train_mask = (panel.index < day0) & valid
        fc_mask = (panel.index >= day0) & (panel.index < day1)
        fc_idx = panel.index[fc_mask]
        if len(fc_idx) != 24:
            continue

        if int(train_mask.sum()) < min_train_hours:
            continue

        X_tr = X.loc[train_mask]
        y_tr = y.loc[train_mask]
        X_fc = X.loc[fc_idx]

        beta = fit_ols_beta(X_tr, y_tr)
        y_reg = pd.Series(predict_ols(X_fc, beta), index=fc_idx, name="regression")

        hist = panel.loc[panel.index < day0, PRICE_COL]
        y_nl = naive_price_forecast(hist, 24, method="last").reindex(fc_idx).rename("naive_last")
        y_ns = naive_price_forecast(hist, 24, method="seasonal_24h").reindex(fc_idx).rename(
            "naive_seasonal_24h"
        )

        y_a = panel.loc[fc_idx, PRICE_COL].rename("actual")

        reg_chunks.append(y_reg)
        naive_last_chunks.append(y_nl)
        naive_seas_chunks.append(y_ns)
        actual_chunks.append(y_a)

        per_day.append(
            {
                "eval_date_utc": eval_d.isoformat(),
                "n_train_hours": int(train_mask.sum()),
                "regression": backtest_report(y_a, y_reg),
                "naive_last": backtest_report(y_a, y_nl),
                "naive_seasonal_24h": backtest_report(y_a, y_ns),
            }
        )

    if not actual_chunks:
        raise ValueError(
            "no completed eval days (increase history, lower min_train_hours, or reduce n_eval_days)"
        )

    act = pd.concat(actual_chunks).sort_index()
    pred_r = pd.concat(reg_chunks).sort_index()
    pred_nl = pd.concat(naive_last_chunks).sort_index()
    pred_ns = pd.concat(naive_seas_chunks).sort_index()

    overall = {
        "regression": backtest_report(act, pred_r),
        "naive_last": backtest_report(act, pred_nl),
        "naive_seasonal_24h": backtest_report(act, pred_ns),
    }

    forecast_frame = pd.DataFrame(
        {
            "actual": act,
            "regression": pred_r,
            "naive_last": pred_nl,
            "naive_seasonal_24h": pred_ns,
        }
    ).sort_index()

    return {
        "bundle_dir": str(bundle_dir.resolve()),
        "n_eval_days_requested": n_eval_days,
        "n_eval_days_completed": len(per_day),
        "min_train_hours": min_train_hours,
        "spec": (
            f"{PRICE_COL} ~ intercept + {', '.join(c for c in X.columns if c != 'intercept')} "
            "(hour ref 0, dow ref Mon)"
        ),
        "overall_metrics": overall,
        "per_day": per_day,
        "forecast_index_start": act.index.min().isoformat(),
        "forecast_index_end": act.index.max().isoformat(),
        "forecast_frame": forecast_frame,
    }
