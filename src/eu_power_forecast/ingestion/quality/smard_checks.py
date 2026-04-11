"""Data quality rules for time series and SMARD DE-LU hourly pulls."""

from __future__ import annotations

from typing import Any

import pandas as pd

from eu_power_forecast.ingestion.smard_data import SmardData
from eu_power_forecast.utils.quality.time_series_checks import (
    evaluate_non_negative_numeric_columns,
    evaluate_timestamp_column_duplicates,
    evaluate_utc_hourly_datetime_index,
)


def basic_time_series_checks(df: pd.DataFrame, ts_col: str) -> dict[str, Any]:
    """True when ``ts_col`` has no duplicate timestamps."""
    return evaluate_timestamp_column_duplicates(df, ts_col)


def check_utc_hourly_index(idx: pd.DatetimeIndex) -> dict[str, Any]:
    """UTC-aware, monotonic, unique, and mostly hourly steps (SMARD index is UTC)."""
    return evaluate_utc_hourly_datetime_index(idx)


def check_non_negative(df: pd.DataFrame, *, atol: float = 0.0) -> dict[str, Any]:
    """Each numeric column has no values below ``-atol``."""
    return evaluate_non_negative_numeric_columns(df, atol=atol)


def qa_de_lu_smard_bundle(
    data: SmardData,
    *,
    require_non_negative_prices: bool = False,
) -> dict[str, Any]:
    """
    DE-LU hourly core tables: UTC hourly index, monotonic unique timestamps, nonnegative checks.

    Day-ahead prices are often negative in the EU; by default we do **not** require prices ≥ 0.
    Set ``require_non_negative_prices=True`` to hard-fail on negative EUR/MWh.
    """
    if not isinstance(data, SmardData):
        raise TypeError("expected SmardData")

    out: dict[str, Any] = {"ok": True, "tables": {}}
    if (data.resolution or "").lower() != "hour":
        out["ok"] = False
        out["bundle_resolution"] = {"expected": "hour", "actual": data.resolution}

    for name, df in data.iter_core_tables():
        if df.empty:
            out["tables"][name] = {"hourly_utc": {"ok": True, "note": "empty_frame"}, "non_negative": {"ok": True}}
            continue
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(idx)
        hourly = check_utc_hourly_index(idx)
        if name == "day_ahead_prices" and not require_non_negative_prices:
            neg: dict[str, Any] = {"ok": True, "skipped": "allow_negative_prices"}
        else:
            neg = check_non_negative(df)
        out["tables"][name] = {"hourly_utc": hourly, "non_negative": neg}
        if not hourly.get("ok", False) or not neg.get("ok", False):
            out["ok"] = False

    return out
