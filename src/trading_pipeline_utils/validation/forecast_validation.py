import logging

import numpy as np
import pandas as pd

from trading_pipeline_utils.settings import ColumnMap, ModelPipelineConfig, ValidationConfig

logger = logging.getLogger(__name__)


def _ensure_utc_index(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    if ts_col in out.columns:
        ts = pd.to_datetime(out[ts_col], utc=True)
        out = out.drop(columns=[ts_col])
        out.index = ts
    elif not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("DataFrame needs DatetimeIndex or timestamp column")
    else:
        out.index = pd.to_datetime(out.index, utc=True)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out.sort_index()
    return out


def _add_local_calendar(out: pd.DataFrame, tz: str, c: ColumnMap) -> pd.DataFrame:
    local = out.index.tz_convert(tz)
    out[f"{c.timestamp}_local"] = local
    out["delivery_hour_local"] = local.hour
    out["delivery_day_of_week_local"] = local.dayofweek
    out["delivery_month_local"] = local.month
    out["is_weekend_local"] = (local.dayofweek >= 5).astype(np.int8)
    out["hour_of_week_local"] = local.dayofweek * 24 + local.hour
    return out


def validate_hourly_frame(df: pd.DataFrame, config: ModelPipelineConfig) -> pd.DataFrame:
    """
    Sort by UTC time, drop duplicate rows, fail on duplicate timestamps,
    optional strict hourly grid, add local delivery-time columns.
    """
    vc: ValidationConfig = config.validation
    c = config.columns
    out = _ensure_utc_index(df, c.timestamp)

    if vc.drop_exact_row_duplicates:
        n0 = len(out)
        out = out[~out.duplicated(keep="first")]
        if len(out) < n0:
            logger.info("Dropped %s exact duplicate rows", n0 - len(out))

    dup_ts = out.index.duplicated(keep=False)
    if dup_ts.any():
        raise ValueError(
            f"Duplicate timestamps after UTC normalisation ({int(dup_ts.sum())} rows). "
            "Resolve upstream before modelling."
        )

    if vc.require_strictly_hourly and len(out) >= 2:
        deltas = out.index.to_series().diff().dropna()
        bad = deltas != pd.Timedelta(hours=1)
        if bad.any():
            first_bad_ts = deltas.loc[bad].index[0]
            msg = (
                f"Non-hourly steps detected ({int(bad.sum())} transitions). "
                f"First issue at {first_bad_ts}."
            )
            if vc.missing_hours_policy == "fail":
                raise ValueError(msg)
            logger.warning("%s — continuing with keep_na policy", msg)

    out = _add_local_calendar(out, vc.delivery_timezone, c)
    return out


def assert_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
