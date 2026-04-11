"""Forecast vs actual errors for wind, solar, and hydro (MW, hourly)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from configs.configs import META_NAME
from eu_power_forecast.ingestion.smard_data import get_smard_index_data

# SMARD realized run-of-river hydro (Wasserkraft), MW — use as "hydro actual" vs ``hydro_forecast`` bundle column.
_FILTER_HYDRO_GENERATION_ACTUAL = "1226"
_HYDRO_ACTUAL_COLUMN = "hydro_generation_actual_mw"


def _one_column_series(df: pd.DataFrame, context: str) -> pd.Series:
    if df.shape[1] != 1:
        raise ValueError(f"{context}: expected exactly one column, got {df.shape[1]}")
    return df.iloc[:, 0].astype("float64")


def fetch_hydro_generation_actual_mw(
    region: str = "DE-LU",
    resolution: str = "hour",
    timestamp: str | None = None,
) -> pd.DataFrame:
    """
    Download realized run-of-river hydro (SMARD filter 1226), same slice as the hourly bundle.

    ``hydro_forecast`` in the bundle is SMARD "Sonstige" prognose (not hydro-only); this series is
    the closest standard **actual** hydro generation counterpart for error diagnostics.
    """
    raw = get_smard_index_data(_FILTER_HYDRO_GENERATION_ACTUAL, region, resolution, timestamp)
    if raw.shape[1] != 1:
        raise ValueError(f"hydro actual {raw.shape=}")
    return raw.rename(columns={raw.columns[0]: _HYDRO_ACTUAL_COLUMN})


def compute_power_forecast_errors_mw(
    wind_generation_actual_mw: pd.DataFrame,
    wind_forecast_mw: pd.DataFrame,
    solar_generation_actual_mw: pd.DataFrame,
    solar_forecast: pd.DataFrame,
    hydro_generation_actual_mw: pd.DataFrame,
    hydro_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forecast errors (MW, actual minus forecast)::

        wind_error  = wind_actual  - wind_forecast
        solar_error = solar_actual - solar_forecast
        hydro_error = hydro_actual - hydro_forecast

    All inputs are **inner-joined** on the time index.
    """
    tbl = pd.concat(
        {
            "wind_a": _one_column_series(wind_generation_actual_mw, "wind_generation_actual_mw"),
            "wind_f": _one_column_series(wind_forecast_mw, "wind_forecast_mw"),
            "solar_a": _one_column_series(solar_generation_actual_mw, "solar_generation_actual_mw"),
            "solar_f": _one_column_series(solar_forecast, "solar_forecast"),
            "hydro_a": _one_column_series(hydro_generation_actual_mw, "hydro_generation_actual_mw"),
            "hydro_f": _one_column_series(hydro_forecast, "hydro_forecast"),
        },
        axis=1,
        join="inner",
    )
    out = pd.DataFrame(
        {
            "wind_error_mw": tbl["wind_a"] - tbl["wind_f"],
            "solar_error_mw": tbl["solar_a"] - tbl["solar_f"],
            "hydro_error_mw": tbl["hydro_a"] - tbl["hydro_f"],
        },
        index=tbl.index,
    )
    return out


def _load_hydro_actual_bundle_path(bundle_dir: Path) -> Path | None:
    p = bundle_dir / f"{_HYDRO_ACTUAL_COLUMN}.parquet"
    return p if p.is_file() else None


def power_forecast_errors_mw_from_bundle_dir(
    bundle_dir: str | Path,
    *,
    hydro_generation_actual_mw: pd.DataFrame | None = None,
    fetch_hydro_if_missing: bool = True,
) -> pd.DataFrame:
    """
    Load wind/solar/forecast Parquets from ``bundle_dir`` and build error columns.

    Hydro actual is taken from (in order): explicit ``hydro_generation_actual_mw``,
    ``hydro_generation_actual_mw.parquet`` in the bundle directory, or — if
    ``fetch_hydro_if_missing`` — a live SMARD pull for filter 1226 (region/resolution from
    ``_smard_bundle_meta.json`` when present).
    """
    d = Path(bundle_dir)
    wind_a = pd.read_parquet(d / "wind_generation_actual_mw.parquet")
    wind_f = pd.read_parquet(d / "wind_forecast_mw.parquet")
    solar_a = pd.read_parquet(d / "solar_generation_actual_mw.parquet")
    solar_f = pd.read_parquet(d / "solar_forecast.parquet")
    hydro_f = pd.read_parquet(d / "hydro_forecast.parquet")

    hydro_a = hydro_generation_actual_mw
    if hydro_a is None:
        path = _load_hydro_actual_bundle_path(d)
        if path is not None:
            hydro_a = pd.read_parquet(path)
    if hydro_a is None:
        if not fetch_hydro_if_missing:
            raise FileNotFoundError(
                f"Missing hydro actual: pass hydro_generation_actual_mw=..., add "
                f"{_HYDRO_ACTUAL_COLUMN}.parquet under {d}, or set fetch_hydro_if_missing=True."
            )
        meta_path = d / META_NAME
        region, resolution = "DE-LU", "hour"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            region = str(meta.get("region", region))
            resolution = str(meta.get("resolution", resolution))
        hydro_a = fetch_hydro_generation_actual_mw(region=region, resolution=resolution)

    return compute_power_forecast_errors_mw(wind_a, wind_f, solar_a, solar_f, hydro_a, hydro_f)
