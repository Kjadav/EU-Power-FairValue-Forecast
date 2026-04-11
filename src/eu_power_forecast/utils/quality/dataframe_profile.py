"""Deterministic dataframe profile for QA and LLM context (no LLM calls)."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_dataframe_profile(df: pd.DataFrame, table_name: str) -> dict[str, Any]:
    """Missingness, duplicate index, and time coverage stats."""
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
        out["time_coverage"] = {"start_utc": idx.min().isoformat(), "end_utc": idx.max().isoformat()}
    else:
        out["time_coverage"] = {"note": "index_not_datetime"}
    return out
