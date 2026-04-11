from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from data.fetcher import DataPayload

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    ok: bool
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)


def build_profile(df: pd.DataFrame, table_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "table": table_name,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "missingness": {str(c): float(df[c].isna().mean()) for c in df.columns},
        "duplicate_index_count": int(df.index.duplicated().sum()),
    }
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        deltas = pd.Series(idx).diff().dropna().dt.total_seconds()
        out["time_coverage"] = {
            "start_utc": idx.min().isoformat(),
            "end_utc": idx.max().isoformat(),
            "median_step_seconds": float(deltas.median()),
            "pct_hourly_steps": float(deltas.between(3599.0, 3601.0).mean()),
        }
    elif isinstance(idx, pd.DatetimeIndex):
        out["time_coverage"] = {
            "start_utc": idx.min().isoformat(),
            "end_utc": idx.max().isoformat(),
        }
    else:
        out["time_coverage"] = {"note": "index_not_datetime"}
    return out


def _is_utc(index: pd.DatetimeIndex) -> bool:
    if index.tz is None:
        return False
    if index.tz == datetime.timezone.utc:
        return True
    return str(index.tz).upper().startswith("UTC")


def _check_utc_hourly(index: pd.DatetimeIndex) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": True}
    if not isinstance(index, pd.DatetimeIndex):
        return {"ok": False, "reason": "index_not_datetime"}
    if not _is_utc(index):
        return {"ok": False, "reason": "not_utc_aware"}
    if not index.is_monotonic_increasing:
        return {"ok": False, "reason": "not_monotonic"}
    if index.has_duplicates:
        return {"ok": False, "reason": "duplicate_timestamps"}
    if len(index) < 2:
        result["median_step_seconds"] = None
        return result
    delta_s = index.to_series().diff().dropna().dt.total_seconds()
    result["median_step_seconds"] = float(delta_s.median())
    result["pct_steps_one_hour"] = float(delta_s.between(3599.0, 3601.0).mean())
    if result["pct_steps_one_hour"] < 0.95:
        result["ok"] = False
        result["reason"] = "not_mostly_hourly"
    return result


def _check_non_negative(df: pd.DataFrame, *, atol: float = 0.0) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": True, "columns": {}}
    for col in df.select_dtypes(include=["number"]).columns:
        n_neg = int((df[col] < -atol).sum())
        result["columns"][col] = {
            "n_negative": n_neg,
            "min": float(df[col].min()) if len(df[col]) else None,
        }
        if n_neg:
            result["ok"] = False
    return result


def validate(data: DataPayload, config: dict[str, Any]) -> ValidationResult:
    """Statistical and structural validation of all core tables."""
    overall_ok = True
    tables: dict[str, dict[str, Any]] = {}
    profiles: dict[str, dict[str, Any]] = {}

    if (data.resolution or "").lower() != "hour":
        overall_ok = False

    for name, df in data.iter_core_tables():
        profiles[name] = build_profile(df, name)
        if df.empty:
            tables[name] = {
                "hourly_utc": {"ok": True, "note": "empty_frame"},
                "non_negative": {"ok": True},
            }
            continue

        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.index)
        hourly = _check_utc_hourly(idx)

        if name == "day_ahead_prices":
            neg: dict[str, Any] = {"ok": True, "skipped": "allow_negative_prices"}
        else:
            neg = _check_non_negative(df)

        tables[name] = {"hourly_utc": hourly, "non_negative": neg}
        if not hourly.get("ok", False) or not neg.get("ok", False):
            overall_ok = False

    return ValidationResult(ok=overall_ok, tables=tables, profiles=profiles)
