from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd

from configs.configs import (
    DEFAULT_PHYSICAL_FLOW_IDS,
    FILTER_GENERATION_PUMPED_STORAGE,
    FILTER_SOLAR_GENERATION_ACTUAL,
    PHYSICAL_FLOW_VALUE_COLUMN,
    REALIZED_GENERATION_CARRIER_FILTERS,
    SMARD_BUNDLE_SINGLE_SERIES,
    SMARD_TABLE_VALUE_COLUMNS,
    SMARD_WIND_FORECAST_SOURCES,
    SMARD_WIND_GENERATION_SOURCES,
)
from eu_power_forecast.ingestion.smard_data import SmardData, get_smard_index_data
from eu_power_forecast.llm_integration.data_fetch.quality.llm_data_qa import qa_smard_bundle_report

logger = logging.getLogger(__name__)


def _fetch_series(
    filter_id: str,
    region: str,
    resolution: str,
    timestamp: str | None = None,
) -> pd.DataFrame:
    return get_smard_index_data(filter_id, region, resolution, timestamp)


def _rename_single_value_column(df: pd.DataFrame, new_name: str) -> pd.DataFrame:
    """SMARD returns one numeric column ``{filter_id}_value``; rename for clarity."""
    if df.shape[1] != 1:
        return df
    return df.rename(columns={df.columns[0]: new_name})


def _wind_onshore_offshore_sum_mw(
    region: str,
    resolution: str,
    sources: tuple[tuple[str, str], ...],
    value_column: str,
) -> pd.DataFrame:
    """Sum onshore + offshore wind series (MW), inner-joined on timestamp."""
    frames: list[pd.DataFrame] = []
    for filter_id, _label in sources:
        raw = _fetch_series(filter_id, region, resolution)
        if raw.shape[1] != 1:
            raise ValueError(f"wind part {filter_id}: expected one column, got {raw.shape}")
        frames.append(raw.rename(columns={raw.columns[0]: filter_id}))
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="inner")
    if merged.shape[1] != len(sources):
        raise ValueError("wind join: column count mismatch")
    return merged.sum(axis=1).to_frame(value_column)


def _wind_forecast_mw(region: str, resolution: str) -> pd.DataFrame:
    return _wind_onshore_offshore_sum_mw(
        region, resolution, SMARD_WIND_FORECAST_SOURCES, "wind_forecast_mw"
    )


def _wind_generation_actual_mw(region: str, resolution: str) -> pd.DataFrame:
    return _wind_onshore_offshore_sum_mw(
        region,
        resolution,
        SMARD_WIND_GENERATION_SOURCES,
        SMARD_TABLE_VALUE_COLUMNS["wind_generation_actual_mw"],
    )


def _actual_generation_total_mw(region: str, resolution: str) -> pd.DataFrame:
    """Sum of realized net generation by carrier (SMARD MW).

    Uses an **outer** join so a missing carrier at a timestamp counts as 0 for that hour; pure inner
    joins can drop all rows if SMARD slices differ slightly between filter ids.
    """
    column = SMARD_TABLE_VALUE_COLUMNS["actual_generation_total_mw"]
    joined: pd.DataFrame | None = None
    for filter_id in REALIZED_GENERATION_CARRIER_FILTERS:
        raw = _fetch_series(filter_id, region, resolution)
        if raw.shape[1] != 1:
            raise ValueError(f"generation {filter_id}: expected one column, got {raw.shape}")
        piece = raw.rename(columns={raw.columns[0]: filter_id})
        joined = piece if joined is None else joined.join(piece, how="outer")
    if joined is None or joined.empty:
        raise ValueError("actual generation: no carrier data")
    return joined.fillna(0.0).sum(axis=1).to_frame(column).sort_index()


def download_smard_data_de_lu(
    output_dir: Path,
    *,
    region: str = "DE-LU",
    resolution: str = "hour",
    load_filter_id: str | None = None,
    physical_flow_resolution: str = "quarterhour",
    physical_flow_ids: Sequence[str] | None = None,
    include_physical_flows: bool = True,
    on_flow_error: Literal["skip", "raise"] = "skip",
    write_qa_report: bool = True,
    llm_qa: bool = False,
    llm_model: str = "gpt-4o-mini",
    qa_min_confidence: float = 0.85,
) -> SmardData:
    """
    Download SMARD Data for bidding zone: DE-LU

    Series pulled from ``configs.configs`` (``SMARD_BUNDLE_SINGLE_SERIES``, wind sources,
    ``REALIZED_GENERATION_CARRIER_FILTERS``, ``FILTER_GENERATION_PUMPED_STORAGE``).

    Args:
        output_dir: directory to save data to
        region: bidding zone : DE-LU
        resolution: hourly resolution for market / generation bundle tables
        load_filter_id: override load forecast SMARD filter (default from config entry ``load_forecast``)
        physical_flow_resolution: resolution for physical flows
        physical_flow_ids: ids for physical flows
        include_physical_flows: whether to include physical flows
        on_flow_error: what to do if a physical flow fails
        write_qa_report: write ``qa_report.json`` (baseline + optional LLM rules) next to Parquet
        llm_qa: if True, call OpenAI to propose whitelist rules (needs ``OPENAI_API_KEY`` and ``openai``)
        llm_model: OpenAI chat model id
        qa_min_confidence: rules below this confidence are reported but do not fail the bundle gate
    Returns:
        SmardData: SMARD data for DE-LU
    """
    output_dir = Path(output_dir)
    physical_flow_ids = tuple(physical_flow_ids) if physical_flow_ids is not None else DEFAULT_PHYSICAL_FLOW_IDS

    series_by_name: dict[str, pd.DataFrame] = {}
    for table_name, (filter_id, _desc) in SMARD_BUNDLE_SINGLE_SERIES.items():
        fid = load_filter_id if table_name == "load_forecast" and load_filter_id is not None else filter_id
        series_by_name[table_name] = _rename_single_value_column(
            _fetch_series(fid, region, resolution),
            SMARD_TABLE_VALUE_COLUMNS[table_name],
        )

    wind_forecast_mw = _wind_forecast_mw(region, resolution)
    wind_generation_actual_mw = _wind_generation_actual_mw(region, resolution)
    solar_generation_actual_mw = _rename_single_value_column(
        _fetch_series(FILTER_SOLAR_GENERATION_ACTUAL, region, resolution),
        SMARD_TABLE_VALUE_COLUMNS["solar_generation_actual_mw"],
    )
    actual_generation_total_mw = _actual_generation_total_mw(region, resolution)
    hydro_pumped_storage_generation_mw = _rename_single_value_column(
        _fetch_series(FILTER_GENERATION_PUMPED_STORAGE, region, resolution),
        SMARD_TABLE_VALUE_COLUMNS["hydro_pumped_storage_generation_mw"],
    )

    physical_flows: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    if include_physical_flows:
        for forecast_id in physical_flow_ids:
            try:
                raw = _fetch_series(forecast_id, region, physical_flow_resolution)
                physical_flows[forecast_id] = _rename_single_value_column(
                    raw, PHYSICAL_FLOW_VALUE_COLUMN
                )
                logger.info("Fetched physical flow %s", forecast_id)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                errors[forecast_id] = msg
                logger.warning("Failed physical flow %s: %s", forecast_id, msg)
                if on_flow_error == "raise":
                    raise

    bundle = SmardData(
        region=region,
        resolution=resolution,
        day_ahead_prices=series_by_name["day_ahead_prices"],
        load_forecast=series_by_name["load_forecast"],
        wind_forecast_mw=wind_forecast_mw,
        solar_forecast=series_by_name["solar_forecast"],
        hydro_forecast=series_by_name["hydro_forecast"],
        wind_generation_actual_mw=wind_generation_actual_mw,
        solar_generation_actual_mw=solar_generation_actual_mw,
        actual_generation_total_mw=actual_generation_total_mw,
        hydro_pumped_storage_generation_mw=hydro_pumped_storage_generation_mw,
        physical_flows=physical_flows,
        physical_flow_errors=errors,
    )
    bundle.save(output_dir)
    if write_qa_report:
        report = qa_smard_bundle_report(
            bundle,
            use_llm=llm_qa,
            model=llm_model,
            min_confidence_to_enforce=qa_min_confidence,
        )
        (output_dir / "qa_report.json").write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(
            "QA report written (llm_qa=%s overall_ok=%s)",
            llm_qa,
            report.get("summary", {}).get("overall_ok"),
        )
    return bundle
