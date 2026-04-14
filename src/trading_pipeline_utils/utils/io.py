from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_parquet_safe(path: Path, label: str) -> pd.DataFrame:
    """Load a Parquet file, raising FileNotFoundError with a clear message if missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Required upstream artifact missing: {label} at {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{label}: expected DatetimeIndex, got {type(df.index)}")
    return df


def load_csv_safe(path: Path, label: str) -> pd.DataFrame:
    """Load a CSV file, returning empty DataFrame if missing."""
    if not path.is_file():
        logger.warning("Optional upstream artifact missing: %s at %s", label, path)
        return pd.DataFrame()
    return pd.read_csv(path)


def load_json_safe(path: Path, label: str) -> dict[str, Any]:
    """Load a JSON file, returning empty dict if missing."""
    if not path.is_file():
        logger.warning("Optional upstream artifact missing: %s at %s", label, path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
