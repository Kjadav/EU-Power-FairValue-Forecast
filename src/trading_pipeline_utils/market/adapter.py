from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_pipeline_utils.settings import PostModelConfig
from trading_pipeline_utils.types import AggregationLevel

logger = logging.getLogger(__name__)


@dataclass
class ForecastSnapshot:
    point_forecasts: pd.DataFrame
    scenario_paths: pd.DataFrame | None
    backtest_summary: dict[str, Any]
    oof_predictions: pd.DataFrame | None
    aggregation_level: AggregationLevel
    forecast_run_time: pd.Timestamp
    as_of_date: str
    delivery_timezone: str
    market_name: str
    market_forwards: dict[str, float] = field(default_factory=dict)


_REQUIRED_POINT_COLS = {"price_mean", "price_p10", "price_p50", "price_p90"}


def _load_parquet_safe(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required upstream artifact missing: {label} at {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{label}: expected DatetimeIndex, got {type(df.index)}")
    return df


def _load_csv_safe(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        logger.warning("Optional upstream artifact missing: %s at %s", label, path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_json_safe(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        logger.warning("Optional upstream artifact missing: %s at %s", label, path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise_point_forecasts(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename upstream price columns to the stable names the rest of the pipeline uses."""
    col_map = {
        "y_point": "price_mean",
        "q10": "price_p10",
        "q50": "price_p50",
        "q90": "price_p90",
    }
    out = raw.rename(columns=col_map)
    missing = _REQUIRED_POINT_COLS - set(out.columns)
    if missing:
        raise ValueError(f"Point forecast missing columns after mapping: {missing}")

    out.index.name = "timestamp_utc"
    return out[sorted(_REQUIRED_POINT_COLS)]


def _normalise_scenario_paths(raw: pd.DataFrame) -> pd.DataFrame:
    path_cols = [c for c in raw.columns if c.startswith("path_")]
    if not path_cols:
        raise ValueError("Scenario paths file has no path_* columns")
    out = raw[path_cols].copy()
    out.index.name = "timestamp_utc"
    return out


def _determine_aggregation_level(
    has_paths: bool,
    cfg: PostModelConfig,
) -> AggregationLevel:
    if has_paths:
        return "scenario_paths"
    if cfg.enable_quantile_approximation:
        return "quantiles_approx"
    return "point_only"


def load_forecast_snapshot(
    artifact_dir: Path,
    config: PostModelConfig,
    *,
    market_forwards: dict[str, float] | None = None,
    forward_csv: Path | None = None,
) -> ForecastSnapshot:
    artifact_dir = Path(artifact_dir)

    raw_fc = _load_parquet_safe(artifact_dir / "final_forecasts.parquet", "final_forecasts")
    point = _normalise_point_forecasts(raw_fc)

    paths_path = artifact_dir / "simulated_paths.parquet"
    scenario_paths: pd.DataFrame | None = None
    has_paths = paths_path.is_file()
    if has_paths:
        raw_paths = _load_parquet_safe(paths_path, "simulated_paths")
        scenario_paths = _normalise_scenario_paths(raw_paths)

    bt_csv = _load_csv_safe(artifact_dir / "backtest_summary.csv", "backtest_summary")
    bt_summary: dict[str, Any] = {}
    if not bt_csv.empty:
        bt_summary = bt_csv.iloc[0].to_dict()

    oof_path = artifact_dir / "oof_predictions.parquet"
    oof: pd.DataFrame | None = None
    if oof_path.is_file():
        oof = pd.read_parquet(oof_path)

    agg = _determine_aggregation_level(has_paths, config)

    fwd: dict[str, float] = dict(market_forwards or {})
    if forward_csv is not None and Path(forward_csv).is_file():
        fwd_df = pd.read_csv(forward_csv)
        for _, row in fwd_df.iterrows():
            fwd[str(row["product_code"])] = float(row["market_forward"])

    now = pd.Timestamp.now(tz="UTC")
    return ForecastSnapshot(
        point_forecasts=point,
        scenario_paths=scenario_paths,
        backtest_summary=bt_summary,
        oof_predictions=oof,
        aggregation_level=agg,
        forecast_run_time=now,
        as_of_date=now.strftime("%Y-%m-%d"),
        delivery_timezone=config.products.delivery_timezone,
        market_name=config.market_name,
        market_forwards=fwd,
    )
