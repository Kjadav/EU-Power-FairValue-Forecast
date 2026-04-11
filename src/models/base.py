from __future__ import annotations

from typing import Literal

import pandas as pd

PriceMethod = Literal["last", "seasonal_24h"]


def _as_price_series(
    prices: pd.Series | pd.DataFrame, value_col: str | None = None
) -> pd.Series:
    if isinstance(prices, pd.DataFrame):
        if value_col and value_col in prices.columns:
            s = prices[value_col]
        else:
            s = prices.iloc[:, 0]
    else:
        s = prices
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("price history must use a DatetimeIndex")
    s = s.sort_index()
    if s.index.has_duplicates:
        s = s[~s.index.duplicated(keep="last")]
    return s.astype("float64")


def _infer_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) >= 2:
        delta = index[-1] - index[-2]
        if delta > pd.Timedelta(0):
            return delta
    return pd.Timedelta(hours=1)


def forecast(
    history: pd.Series | pd.DataFrame,
    horizon: int,
    *,
    method: PriceMethod = "last",
    value_col: str | None = None,
) -> pd.Series:
    """Naive baselines for hourly day-ahead prices (EUR/MWh)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    s = _as_price_series(history, value_col)
    if s.empty:
        raise ValueError("history is empty")

    step = _infer_step(s.index)
    start = s.index[-1] + step
    future = pd.date_range(start=start, periods=horizon, freq=step, tz=s.index[-1].tz)
    name = s.name or "price_eur_mwh"

    if method == "last":
        return pd.Series(float(s.iloc[-1]), index=future, name=name)

    lag = pd.Timedelta(hours=24)
    values: list[float] = []
    for ts in future:
        v = s.asof(ts - lag)
        values.append(float(v) if not pd.isna(v) else float(s.iloc[-1]))
    return pd.Series(values, index=future, dtype="float64", name=name)
