"""Design-matrix helpers for day-ahead price regression (bundle-aligned, reproducible)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from configs.configs import SMARD_TABLE_VALUE_COLUMNS
from eu_power_forecast.forecasting.residual_load import residual_load_mw_from_bundle_dir

PRICE_COL = SMARD_TABLE_VALUE_COLUMNS["day_ahead_prices"]
HYDRO_FORECAST_COL = SMARD_TABLE_VALUE_COLUMNS["hydro_forecast"]
RESIDUAL_COL = "residual_load_mw"


def load_day_ahead_regression_panel(bundle_dir: str | Path) -> pd.DataFrame:
    """Inner-join day-ahead price, residual load (from forecasts), and hydro forecast (MW)."""
    d = Path(bundle_dir)
    prices = pd.read_parquet(d / "day_ahead_prices.parquet")
    residual = residual_load_mw_from_bundle_dir(d)
    hydro = pd.read_parquet(d / "hydro_forecast.parquet")
    if prices.shape[1] != 1 or hydro.shape[1] != 1:
        raise ValueError("expected single-column day_ahead_prices and hydro_forecast frames")
    p = prices.iloc[:, 0].rename(PRICE_COL)
    h = hydro.iloc[:, 0].rename(HYDRO_FORECAST_COL)
    out = pd.concat([p, residual[RESIDUAL_COL], h], axis=1, join="inner").sort_index()
    return out


def add_autoregressive_price_lags(panel: pd.DataFrame) -> pd.DataFrame:
    """Add ``price_lag_24`` and ``price_lag_168`` from ``PRICE_COL``."""
    out = panel.copy()
    out["price_lag_24"] = out[PRICE_COL].shift(24)
    out["price_lag_168"] = out[PRICE_COL].shift(168)
    return out


def time_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour (ref 0) and weekday (ref Monday) dummies; fixed category sets for stable columns."""
    hour = pd.Categorical(index.hour, categories=list(range(24)))
    dow = pd.Categorical(index.dayofweek, categories=list(range(7)))
    h = pd.get_dummies(hour, prefix="h", drop_first=True, dtype=float)
    d = pd.get_dummies(dow, prefix="dow", drop_first=True, dtype=float)
    h.index = index
    d.index = index
    return pd.concat([h, d], axis=1)


def build_design_xy(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    P_t ~ intercept + RL + hydro_fc + P_{t-24} + P_{t-168} + hour + dow dummies.
    """
    base = panel[[RESIDUAL_COL, HYDRO_FORECAST_COL, "price_lag_24", "price_lag_168"]].astype(float)
    td = time_dummies(panel.index)
    intercept = pd.Series(1.0, index=panel.index, name="intercept")
    X = pd.concat([intercept, base, td], axis=1).astype(float)
    y = panel[PRICE_COL].astype(float)
    return X, y


def dates_with_full_24h(index: pd.DatetimeIndex) -> list:
    """Calendar dates (UTC) that have exactly 24 hourly rows."""
    one = pd.Series(np.ones(len(index), dtype=np.int64), index=index)
    by_day = one.groupby(one.index.date).sum()
    return sorted(by_day[by_day == 24].index.tolist())


def row_mask_valid_xy(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    return X.notna().all(axis=1) & y.notna()
