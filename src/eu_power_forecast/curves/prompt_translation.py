"""Translate between curve conventions (zones, products, prompt vs forward)."""

from typing import Any

import pandas as pd


def translate_curve(
    df: pd.DataFrame,
    source_convention: str,
    target_convention: str,
) -> pd.DataFrame:
    """Placeholder: implement mapping tables and calendar logic here."""
    _ = (source_convention, target_convention)
    return df.copy()
