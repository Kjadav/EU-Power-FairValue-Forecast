from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.fetcher import DataPayload

logger = logging.getLogger(__name__)

PRICE_COL = "day_ahead_price_eur_mwh"
HYDRO_FORECAST_COL = "hydro_forecast_mw"
RESIDUAL_COL = "residual_load_mw"


# ---------------------------------------------------------------------------
# residual load
# ---------------------------------------------------------------------------

def _one_col(df: pd.DataFrame, context: str) -> pd.Series:
    if df.shape[1] != 1:
        raise ValueError(f"{context}: expected one column, got {df.shape[1]}")
    return df.iloc[:, 0].astype("float64")


def compute_residual_load(
    load_forecast: pd.DataFrame,
    wind_forecast: pd.DataFrame,
    solar_forecast: pd.DataFrame,
    hydro_forecast: pd.DataFrame,
) -> pd.DataFrame:
    tbl = pd.concat(
        {
            "load": _one_col(load_forecast, "load"),
            "wind": _one_col(wind_forecast, "wind"),
            "solar": _one_col(solar_forecast, "solar"),
            "hydro": _one_col(hydro_forecast, "hydro"),
        },
        axis=1,
        join="inner",
    )
    return (tbl["load"] - tbl["wind"] - tbl["solar"] - tbl["hydro"]).to_frame(RESIDUAL_COL)


def residual_load_from_payload(data: DataPayload) -> pd.DataFrame:
    return compute_residual_load(
        data.load_forecast, data.wind_forecast_mw,
        data.solar_forecast, data.hydro_forecast,
    )


def residual_load_from_bundle(bundle_dir: str | Path) -> pd.DataFrame:
    d = Path(bundle_dir)
    return compute_residual_load(
        pd.read_parquet(d / "load_forecast.parquet"),
        pd.read_parquet(d / "wind_forecast_mw.parquet"),
        pd.read_parquet(d / "solar_forecast.parquet"),
        pd.read_parquet(d / "hydro_forecast.parquet"),
    )


# ---------------------------------------------------------------------------
# panel + features
# ---------------------------------------------------------------------------

def load_regression_panel(bundle_dir: str | Path) -> pd.DataFrame:
    d = Path(bundle_dir)
    prices = pd.read_parquet(d / "day_ahead_prices.parquet")
    residual = residual_load_from_bundle(d)
    hydro = pd.read_parquet(d / "hydro_forecast.parquet")
    p = prices.iloc[:, 0].rename(PRICE_COL)
    h = hydro.iloc[:, 0].rename(HYDRO_FORECAST_COL)
    return pd.concat([p, residual[RESIDUAL_COL], h], axis=1, join="inner").sort_index()


def build_panel_from_payload(data: DataPayload) -> pd.DataFrame:
    residual = residual_load_from_payload(data)
    p = data.day_ahead_prices.iloc[:, 0].rename(PRICE_COL)
    h = data.hydro_forecast.iloc[:, 0].rename(HYDRO_FORECAST_COL)
    return pd.concat([p, residual[RESIDUAL_COL], h], axis=1, join="inner").sort_index()


def add_price_lags(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["price_lag_24"] = out[PRICE_COL].shift(24)
    out["price_lag_168"] = out[PRICE_COL].shift(168)
    return out


def time_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = pd.Categorical(index.hour, categories=list(range(24)))
    dow = pd.Categorical(index.dayofweek, categories=list(range(7)))
    h = pd.get_dummies(hour, prefix="h", drop_first=True, dtype=float)
    d = pd.get_dummies(dow, prefix="dow", drop_first=True, dtype=float)
    h.index = index
    d.index = index
    return pd.concat([h, d], axis=1)


def build_design_xy(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    base = panel[
        [RESIDUAL_COL, HYDRO_FORECAST_COL, "price_lag_24", "price_lag_168"]
    ].astype(float)
    td = time_dummies(panel.index)
    intercept = pd.Series(1.0, index=panel.index, name="intercept")
    X = pd.concat([intercept, base, td], axis=1).astype(float)
    y = panel[PRICE_COL].astype(float)
    return X, y


def dates_with_full_24h(index: pd.DatetimeIndex) -> list:
    one = pd.Series(np.ones(len(index), dtype=np.int64), index=index)
    by_day = one.groupby(one.index.date).sum()
    return sorted(by_day[by_day == 24].index.tolist())


def row_mask_valid(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    return X.notna().all(axis=1) & y.notna()


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------

def fit_ols(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        rcond=None,
    )
    return beta


def predict_ols(X: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ beta


# ---------------------------------------------------------------------------
# forecast errors
# ---------------------------------------------------------------------------

def compute_forecast_errors(
    wind_actual: pd.DataFrame,
    wind_forecast: pd.DataFrame,
    solar_actual: pd.DataFrame,
    solar_forecast: pd.DataFrame,
    hydro_actual: pd.DataFrame,
    hydro_forecast: pd.DataFrame,
) -> pd.DataFrame:
    tbl = pd.concat(
        {
            "wa": _one_col(wind_actual, "wind_actual"),
            "wf": _one_col(wind_forecast, "wind_forecast"),
            "sa": _one_col(solar_actual, "solar_actual"),
            "sf": _one_col(solar_forecast, "solar_forecast"),
            "ha": _one_col(hydro_actual, "hydro_actual"),
            "hf": _one_col(hydro_forecast, "hydro_forecast"),
        },
        axis=1,
        join="inner",
    )
    return pd.DataFrame(
        {
            "wind_error_mw": tbl["wa"] - tbl["wf"],
            "solar_error_mw": tbl["sa"] - tbl["sf"],
            "hydro_error_mw": tbl["ha"] - tbl["hf"],
        },
        index=tbl.index,
    )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def backtest_report(actual: pd.Series, forecast_series: pd.Series) -> dict[str, Any]:
    aligned = pd.concat(
        [actual, forecast_series], axis=1, keys=["y", "y_hat"]
    ).dropna()
    err = aligned["y"] - aligned["y_hat"]
    return {
        "n": len(aligned),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }
