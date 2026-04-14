from __future__ import annotations

from pathlib import Path

import pandas as pd


def _one_column_series(df: pd.DataFrame, context: str) -> pd.Series:
    if df.shape[1] != 1:
        raise ValueError(f"{context}: expected exactly one column, got {df.shape[1]}")
    return df.iloc[:, 0].astype("float64")


def compute_residual_load_mw(
    load_forecast: pd.DataFrame,
    wind_forecast_mw: pd.DataFrame,
    solar_forecast: pd.DataFrame,
    hydro_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """
    Residual load (MW) fromday-ahead forecasts::

        residual_load = load_forecast - (wind_forecast + solar_forecast + hydro_forecast)

    """
    tbl = pd.concat(
        {
            "load": _one_column_series(load_forecast, "load_forecast"),
            "wind_fc": _one_column_series(wind_forecast_mw, "wind_forecast_mw"),
            "solar_fc": _one_column_series(solar_forecast, "solar_forecast"),
            "hydro_fc": _one_column_series(hydro_forecast, "hydro_forecast"),
        },
        axis=1,
        join="inner",
    )
    residual = tbl["load"] - tbl["wind_fc"] - tbl["solar_fc"] - tbl["hydro_fc"]
    #innerjoining
    return residual.to_frame("residual_load_mw")


def residual_load_mw_from_bundle_dir(bundle_dir: str | Path) -> pd.DataFrame:
    d = Path(bundle_dir)
    return compute_residual_load_mw(
        pd.read_parquet(d / "load_forecast.parquet"),
        pd.read_parquet(d / "wind_forecast_mw.parquet"),
        pd.read_parquet(d / "solar_forecast.parquet"),
        pd.read_parquet(d / "hydro_forecast.parquet"),
    )
