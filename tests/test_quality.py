import pandas as pd

from data.validator import build_profile


def test_timestamp_duplicates_none():
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=3, freq="h"), "v": [1, 2, 3]})
    df = df.set_index("ts")
    profile = build_profile(df, "test")
    assert profile["duplicate_index_count"] == 0


def test_timestamp_duplicates_detected():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
    df = pd.DataFrame({"v": [1, 2]}, index=idx)
    profile = build_profile(df, "test")
    assert profile["duplicate_index_count"] == 1
