"""Build the user message payload (schema + sample + profile) for LLM quality proposals."""

from __future__ import annotations

import json

import pandas as pd

from eu_power_forecast.llm_integration.data_fetch.quality.business_context import business_context_block
from eu_power_forecast.utils.quality import build_dataframe_profile


def build_llm_dataset_text_block(
    dataframe: pd.DataFrame,
    table_name: str,
    *,
    sample_row_count: int = 10,
) -> str:
    """Business context, schema, CSV sample, and deterministic profile JSON."""
    columns = [str(c) for c in dataframe.columns]
    index = dataframe.index
    sections: list[str] = [
        business_context_block(table_name, columns),
        "",
        "---",
        "",
        f"dataset_id: {table_name}",
        f"n_rows: {len(dataframe)}",
        f"columns: {columns}",
        f"index_type: {type(index).__name__}",
    ]
    if isinstance(index, pd.DatetimeIndex):
        sections.append(f"index_tz: {index.tz}")
        sections.append(f"index_range: {index.min()} .. {index.max()}")
    sections.append(f"dtypes: {json.dumps({c: str(dataframe[c].dtype) for c in dataframe.columns})}")
    sections.append("sample_csv:\n" + dataframe.head(sample_row_count).to_csv())
    profile = build_dataframe_profile(dataframe, table_name)
    sections.append("profile_json:\n" + json.dumps(profile, indent=2, default=str))
    return "\n".join(sections)
