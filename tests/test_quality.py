import pandas as pd

from trading_pipeline_utils.validation.checks import evaluate_timestamp_column_duplicates


def test_basic_time_series_checks_ok() -> None:
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=3, freq="h"), "v": [1, 2, 3]})
    r = evaluate_timestamp_column_duplicates(df, "ts")
    assert r["ok"] is True


def test_basic_time_series_checks_dupes() -> None:
    df = pd.DataFrame({"ts": ["2024-01-01", "2024-01-01"], "v": [1, 2]})
    r = evaluate_timestamp_column_duplicates(df, "ts")
    assert r["ok"] is False
    assert r["duplicate_timestamps"] == 1
