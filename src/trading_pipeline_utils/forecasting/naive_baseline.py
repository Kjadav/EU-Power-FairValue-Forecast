from __future__ import annotations

from typing import Literal

import pandas as pd

PriceMethod = Literal["last", "seasonal_24h"]


def _as_price_series(
    prices: pd.Series | pd.DataFrame,
    value_col: str | None,
) -> pd.Series:
    if isinstance(prices, pd.DataFrame):
        if value_col is not None:
            if value_col not in prices.columns:
                raise KeyError(f"value_col {value_col!r} not in frame columns")
            s = prices[value_col]
        else:
            s = prices.iloc[:, 0]
    else:
        s = prices
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("price history must use a DatetimeIs")
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


def _future_index(
    last_ts: pd.Timestamp,
    horizon: int,
    step: pd.Timedelta,
) -> pd.DatetimeIndex:
    start = last_ts + step
    return pd.date_range(
        start=start,
        periods=horizon,
        freq=step,
        tz=last_ts.tz,
    )


def naive_price_forecast(
    history: pd.Series | pd.DataFrame,
    horizon: int,
    *,
    method: PriceMethod = "last",
    value_col: str | None = None,
    step: pd.Timedelta | None = None,
) -> pd.Series:
    """
    NAVIE baseline model is createed for market : de-lu
    How does it work?
    Looking a the pricing at the same day and hour of previous day and we compare the prices
    Naive baselines for DE-LU-style hourly day-ahead prices to judge the similarity and see if the model can predict the prices
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    s = _as_price_series(history, value_col)
    if s.empty:
        raise ValueError("history is empty")

    step_td = step if step is not None else _infer_step(s.index)
    future = _future_index(s.index[-1], horizon, step_td)
    name = s.name if s.name is not None else "price_eur_mwh"

    if method == "last":
        value = float(s.iloc[-1])
        return pd.Series(value, index=future, name=name)

    lag = pd.Timedelta(hours=24)
    values: list[float] = []
    for ts in future:
        target = ts - lag
        v = s.asof(target)
        if pd.isna(v):
            v = float(s.iloc[-1])
        else:
            v = float(v)
        values.append(v)
    return pd.Series(values, index=future, dtype="float64", name=name)


def naive_baseline(history: pd.Series, horizon: int) -> pd.Series:
    """last-value naive model is ran"""
    return naive_price_forecast(history, horizon, method="last")
