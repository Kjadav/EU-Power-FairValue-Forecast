"""Parse and normalize LLM rule ``condition`` payloads."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

RANGE_INEQUALITY_PATTERN = re.compile(r"^\s*(>=|<=|>|<)\s*([-+eE0-9.]+)\s*$")


def condition_to_mapping(condition: Any) -> dict[str, Any]:
    if condition is None or condition == "":
        return {}
    if isinstance(condition, dict):
        return condition
    if isinstance(condition, str):
        try:
            return json.loads(condition)
        except json.JSONDecodeError:
            return {}
    return {}


def parse_range_inequality(condition: Any) -> tuple[str, float]:
    if not isinstance(condition, str):
        raise ValueError("range condition must be a string like '>= 0'")
    match = RANGE_INEQUALITY_PATTERN.match(condition.strip())
    if not match:
        raise ValueError(f"bad range condition: {condition!r}")
    return match.group(1), float(match.group(2))


def count_numeric_range_violations(series: pd.Series, operator: str, threshold: float) -> int:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if operator == ">=":
        violations = values < threshold
    elif operator == "<=":
        violations = values > threshold
    elif operator == ">":
        violations = values <= threshold
    elif operator == "<":
        violations = values >= threshold
    else:
        raise ValueError(operator)
    return int(violations.sum())


def require_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("expected DatetimeIndex")
    return idx
