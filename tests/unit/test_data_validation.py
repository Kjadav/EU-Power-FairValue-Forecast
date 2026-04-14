"""Tests for data quality checks: timestamps, profiles, non-negativity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_pipeline_utils.validation.checks import (
    evaluate_timestamp_column_duplicates,
    is_utc_datetime_index,
    evaluate_utc_hourly_datetime_index,
    evaluate_non_negative_numeric_columns,
    build_dataframe_profile,
)


def test_utc_check_positive():
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    assert is_utc_datetime_index(idx)


def test_utc_check_negative():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    assert not is_utc_datetime_index(idx)


def test_hourly_check():
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    result = evaluate_utc_hourly_datetime_index(idx)
    assert result["ok"] is True


def test_non_negative():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [-1.0, 2.0, 3.0]})
    result = evaluate_non_negative_numeric_columns(df)
    assert result["ok"] is False


def test_profile():
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({"val": np.random.randn(10)}, index=idx)
    prof = build_dataframe_profile(df, "test")
    assert prof["table"] == "test"
    assert prof["n_rows"] == 10


def test_duplicate_timestamps():
    df = pd.DataFrame({"ts": ["2024-01-01", "2024-01-01", "2024-01-02"], "v": [1, 2, 3]})
    result = evaluate_timestamp_column_duplicates(df, "ts")
    assert result["ok"] is False
