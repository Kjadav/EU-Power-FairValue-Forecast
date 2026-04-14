import datetime
from typing import Any

import pandas as pd

from trading_pipeline_utils.data.schemas import DataPayload, SmardData, ValidationResult
from trading_pipeline_utils.settings import PipelineConfig

def evaluate_timestamp_column_duplicates(df: pd.DataFrame, timestamp_column: str) -> dict[str, Any]:
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

def build_dataframe_profile(df: pd.DataFrame, table_name: str) -> dict[str, Any]:
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

def build_table_profile(df: pd.DataFrame, table_name: str) -> dict[str, Any]:
    return build_dataframe_profile(df, table_name)


def validate_smard_bundle(
    data: SmardData,
    _config: PipelineConfig,
    *,
    require_non_negative_prices: bool = False,
) -> ValidationResult:
    tables_out: dict[str, dict[str, Any]] = {}
    all_ok = True
    resolution_ok = (data.resolution or "").lower() == "hour"
    if not resolution_ok:
        all_ok = False
        bundle_res: dict[str, Any] = {"ok": False, "expected": "hour", "actual": data.resolution}
    else:
        bundle_res = {"ok": True}

    for name, df in data.iter_core_tables():
        prof = build_table_profile(df, name)
        if df.empty:
            tables_out[name] = {
                "profile": prof,
                "hourly_utc": {"ok": True, "note": "empty_frame"},
                "non_negative": {"ok": True},
            }
            continue
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(idx)
        hourly = evaluate_utc_hourly_datetime_index(idx)
        if name == "day_ahead_prices" and not require_non_negative_prices:
            neg: dict[str, Any] = {"ok": True, "skipped": "allow_negative_prices"}
        else:
            neg = evaluate_non_negative_numeric_columns(df, atol=0.0)
        ok_t = bool(hourly.get("ok", False)) and bool(neg.get("ok", False))
        if not ok_t:
            all_ok = False
        tables_out[name] = {"profile": prof, "hourly_utc": hourly, "non_negative": neg}

    summary = {
        "bundle_resolution_ok": bundle_res,
        "row_counts": {n: len(d) for n, d in data.iter_core_tables()},
        "tables_failed": [n for n, t in tables_out.items() if not _table_checks_ok(t)],
    }
    return ValidationResult(ok=all_ok and resolution_ok, tables=tables_out, summary=summary)


def _table_checks_ok(block: dict[str, Any]) -> bool:
    h = block.get("hourly_utc") or {}
    n = block.get("non_negative") or {}
    return bool(h.get("ok", False)) and bool(n.get("ok", False))


def validate_data_payload(payload: DataPayload, config: PipelineConfig) -> ValidationResult:
    return validate_smard_bundle(payload.bundle, config)


def attach_validation(payload: DataPayload, result: ValidationResult) -> None:
    payload.validation = result


def validation_result_to_jsonable(result: ValidationResult) -> dict[str, Any]:
    return {"ok": result.ok, "tables": result.tables, "summary": result.summary}
