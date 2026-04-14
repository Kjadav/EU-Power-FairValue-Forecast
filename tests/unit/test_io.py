"""Tests for safe file I/O helpers."""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path

import pandas as pd

from trading_pipeline_utils.utils.io import load_parquet_safe, load_csv_safe, load_json_safe


def test_load_parquet_missing():
    with pytest.raises(FileNotFoundError, match="Required upstream artifact"):
        load_parquet_safe(Path("/nonexistent.parquet"), "test")


def test_load_csv_missing():
    df = load_csv_safe(Path("/nonexistent.csv"), "test")
    assert df.empty


def test_load_json_missing():
    d = load_json_safe(Path("/nonexistent.json"), "test")
    assert d == {}
