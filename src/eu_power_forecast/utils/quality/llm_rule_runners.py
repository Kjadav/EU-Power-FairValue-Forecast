"""Execute one LLM-proposed rule against a dataframe."""

from __future__ import annotations

from typing import Any

import pandas as pd

from eu_power_forecast.utils.quality.llm_rule_conditions import (
    condition_to_mapping,
    count_numeric_range_violations,
    parse_range_inequality,
    require_datetime_index,
)


def run_not_null_rule(df: pd.DataFrame, column: str, condition: Any) -> dict[str, Any]:
    if column == "index":
        return {"ok": False, "error": "not_null applies to value columns, not index"}
    if column not in df.columns:
        return {"ok": False, "error": f"missing column {column}"}
    mapping = condition_to_mapping(condition)
    max_null_fraction = float(mapping.get("max_null_fraction", 0.0))
    null_fraction = float(df[column].isna().mean())
    return {
        "ok": null_fraction <= max_null_fraction,
        "null_fraction": null_fraction,
        "max_null_fraction": max_null_fraction,
    }


def run_range_rule(df: pd.DataFrame, column: str, condition: Any) -> dict[str, Any]:
    if column not in df.columns:
        return {"ok": False, "error": f"missing column {column}"}
    operator, threshold = parse_range_inequality(condition)
    violations = count_numeric_range_violations(df[column], operator, threshold)
    return {"ok": violations == 0, "violations": violations, "operator": operator, "threshold": threshold}


def run_frequency_check_rule(df: pd.DataFrame, column: str, condition: Any) -> dict[str, Any]:
    if column != "index":
        return {"ok": False, "error": "frequency_check requires column == 'index'"}
    idx = require_datetime_index(df)
    mapping = condition_to_mapping(condition)
    expected_step = float(mapping.get("expected_step_seconds", 3600))
    tolerance = float(mapping.get("tolerance_seconds", 120.0))
    min_fraction = float(mapping.get("min_fraction_within_tolerance", 0.95))
    if len(idx) < 2:
        return {"ok": True, "note": "too_few_points"}
    deltas = idx.to_series().diff().dropna().dt.total_seconds()
    within = ((deltas - expected_step).abs() <= tolerance).mean()
    return {"ok": float(within) >= min_fraction, "fraction_within_tolerance": float(within)}


def run_no_missing_timestamps_rule(df: pd.DataFrame, column: str, condition: Any) -> dict[str, Any]:
    if column != "index":
        return {"ok": False, "error": "no_missing_timestamps requires column == 'index'"}
    idx = require_datetime_index(df)
    mapping = condition_to_mapping(condition)
    max_gap = float(mapping.get("max_gap_seconds", 7200))
    if len(idx) < 2:
        return {"ok": True, "note": "too_few_points"}
    deltas = idx.to_series().diff().dropna().dt.total_seconds()
    bad_gaps = int((deltas > max_gap).sum())
    return {"ok": bad_gaps == 0, "large_gap_count": bad_gaps}


def run_monotonic_time_index_rule(df: pd.DataFrame, column: str, condition: Any) -> dict[str, Any]:
    _ = condition
    if column != "index":
        return {"ok": False, "error": "monotonic_time_index requires column == 'index'"}
    idx = require_datetime_index(df)
    dupes = int(idx.duplicated().sum())
    mono = bool(idx.is_monotonic_increasing)
    return {"ok": mono and dupes == 0, "is_monotonic": mono, "duplicate_index_count": dupes}


def run_outlier_rule(df: pd.DataFrame, column: str, condition: Any) -> dict[str, Any]:
    if column == "index" or column not in df.columns:
        return {"ok": False, "error": "outlier needs an existing value column"}
    mapping = condition_to_mapping(condition)
    z_threshold = float(mapping.get("z_threshold", 4.0))
    max_outlier_fraction = float(mapping.get("max_outlier_fraction", 0.05))
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(series) < 10:
        return {"ok": True, "note": "too_few_points"}
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std == 0.0:
        return {"ok": True, "note": "constant_series"}
    z_scores = (series - mean) / std
    outlier_fraction = float((z_scores.abs() > z_threshold).mean())
    return {"ok": outlier_fraction <= max_outlier_fraction, "outlier_fraction": outlier_fraction}


def run_volatility_jump_rule(df: pd.DataFrame, column: str, condition: Any) -> dict[str, Any]:
    if column == "index" or column not in df.columns:
        return {"ok": False, "error": "volatility_jump needs an existing value column"}
    mapping = condition_to_mapping(condition)
    cap = float(mapping.get("max_abs_change", 1e12))
    series = pd.to_numeric(df[column], errors="coerce")
    step = series.diff().abs().max()
    if pd.isna(step):
        return {"ok": True, "note": "no_pairs", "max_abs_step_observed": None, "max_abs_change_cap": cap}
    step_f = float(step)
    return {
        "ok": step_f <= cap,
        "max_abs_step_observed": step_f,
        "max_abs_change_cap": cap,
    }
