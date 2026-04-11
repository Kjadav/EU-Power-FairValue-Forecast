import pandas as pd

from eu_power_forecast.ingestion.quality import basic_time_series_checks


def test_basic_time_series_checks_ok():
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=3, freq="h"), "v": [1, 2, 3]})
    r = basic_time_series_checks(df, "ts")
    assert r["ok"] is True


def test_basic_time_series_checks_dupes():
    df = pd.DataFrame({"ts": ["2024-01-01", "2024-01-01"], "v": [1, 2]})
    r = basic_time_series_checks(df, "ts")
    assert r["ok"] is False
    assert r["duplicate_timestamps"] == 1
