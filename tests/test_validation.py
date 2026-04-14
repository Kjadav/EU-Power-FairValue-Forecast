"""Lightweight forecast error sanity checks."""

from __future__ import annotations

import pandas as pd


def test_point_forecast_mae_rmse() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h")
    actual = pd.Series([10.0, 12.0, 11.0, 9.0], index=idx)
    forecast = pd.Series([10.0, 11.0, 12.0, 8.0], index=idx)
    err = actual - forecast
    mae = float(err.abs().mean())
    rmse = float((err**2).mean() ** 0.5)
    assert len(actual) == 4
    assert mae >= 0
    assert rmse >= 0
