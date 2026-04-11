import pandas as pd

from models.forecaster import backtest_report


def test_backtest_report():
    idx = pd.date_range("2024-01-01", periods=4, freq="h")
    actual = pd.Series([10.0, 12.0, 11.0, 9.0], index=idx)
    forecast = pd.Series([10.0, 11.0, 12.0, 8.0], index=idx)
    r = backtest_report(actual, forecast)
    assert r["n"] == 4
    assert r["mae"] >= 0
    assert r["rmse"] >= 0
