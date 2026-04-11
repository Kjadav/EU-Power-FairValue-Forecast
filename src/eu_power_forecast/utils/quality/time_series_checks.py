"""Reusable time-series QA primitives (SMARD and other hourly UTC series)."""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd


def evaluate_timestamp_column_duplicates(df: pd.DataFrame, timestamp_column: str) -> dict[str, Any]:
    """True if ``timestamp_column`` has no duplicate values."""
    if timestamp_column not in df.columns:
        return {"ok": False, "error": f"missing column {timestamp_column}"}
    dupes = int(df[timestamp_column].duplicated().sum())
    return {"ok": dupes == 0, "duplicate_timestamps": dupes}


def is_utc_datetime_index(index: pd.DatetimeIndex) -> bool:
    if index.tz is None:
        return False
    if index.tz == datetime.timezone.utc:
        return True
    tz_name = str(index.tz).upper()
    return tz_name.startswith("UTC") or tz_name == "UTC+00:00"


def evaluate_utc_hourly_datetime_index(index: pd.DatetimeIndex) -> dict[str, Any]:
    """UTC-aware, monotonic, unique, and mostly 1h steps."""
    result: dict[str, Any] = {"ok": True}
    if not isinstance(index, pd.DatetimeIndex):
        return {"ok": False, "reason": "index_not_datetime"}
    if not is_utc_datetime_index(index):
        result["ok"] = False
        result["reason"] = "not_utc_aware"
        return result
    if not index.is_monotonic_increasing:
        result["ok"] = False
        result["reason"] = "not_monotonic"
        return result
    if index.has_duplicates:
        result["ok"] = False
        result["reason"] = "duplicate_timestamps"
        return result
    if len(index) < 2:
        result["median_step_seconds"] = None
        return result
    delta_seconds = index.to_series().diff().dropna().dt.total_seconds()
    median_step = float(delta_seconds.median())
    result["median_step_seconds"] = median_step
    result["pct_steps_one_hour"] = float(delta_seconds.between(3599.0, 3601.0).mean())
    if result["pct_steps_one_hour"] < 0.95:
        result["ok"] = False
        result["reason"] = "not_mostly_hourly"
    return result


def evaluate_non_negative_numeric_columns(df: pd.DataFrame, *, atol: float = 0.0) -> dict[str, Any]:
    """Each numeric column must have no values below ``-atol``."""
    result: dict[str, Any] = {"ok": True, "columns": {}}
    for column in df.select_dtypes(include=["number"]).columns:
        series = df[column]
        negative_count = int((series < -atol).sum())
        result["columns"][column] = {
            "n_negative": negative_count,
            "min": float(series.min()) if len(series) else None,
        }
        if negative_count:
            result["ok"] = False
    return result
