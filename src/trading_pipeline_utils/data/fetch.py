from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

import pandas as pd
import requests

from .schemas import DataPayload, SmardData
from ..settings import DataVendorConfig, PipelineConfig

logger = logging.getLogger(__name__)
T = TypeVar("T")


def fetch_smard_archive_index(
    base_url: str, filter_id: str, region: str, resolution: str
) -> Any:
    """
    Download the JSON index that lists all published archive timestamps for one series
    """
    url = f"{base_url.rstrip('/')}/{filter_id}/{region}/index_{resolution}.json"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def _index_timestamp_list(raw: object) -> list[Any]:
    if isinstance(raw, dict) and "timestamps" in raw:
        return list(raw["timestamps"])
    if isinstance(raw, list):
        return raw
    raise TypeError("SMARD index JSON must be a list or {\"timestamps\": [...]}")


def fetch_smard_archive_chunk(
    base_url: str,
    filter_id: str,
    region: str,
    resolution: str,
    timestamp: str | None = None,
) -> pd.DataFrame:
    """
    Pull SMARD and transform to df.
    """
    if timestamp is None:
        idx_raw = fetch_smard_archive_index(base_url, filter_id, region, resolution)
        ts_list = _index_timestamp_list(idx_raw)
        if not ts_list:
            raise ValueError("SMARD index is empty; cannot resolve latest timestamp")
        timestamp = str(ts_list[-1])

    url = (
        f"{base_url.rstrip('/')}/{filter_id}/{region}/"
        f"{filter_id}_{region}_{resolution}_{timestamp}.json"
    )
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("data") or data.get("series") or []
    df = pd.DataFrame(rows, columns=["unix_ts", "value"])
    df["timestamp"] = pd.to_datetime(df["unix_ts"], unit="ms", utc=True)
    df = df.set_index("timestamp").drop(columns=["unix_ts"])
    return df.rename(columns={"value": f"{filter_id}_value"})


def fetch_smard_series_stitched(
    base_url: str,
    filter_id: str,
    region: str,
    resolution: str,
    *,
    min_hours: int,
    max_index_chunks: int,
    request_pause_s: float = 0.03,
) -> pd.DataFrame:
    idx_raw = fetch_smard_archive_index(base_url, filter_id, region, resolution)
    ts_list = _index_timestamp_list(idx_raw)
    if not ts_list:
        raise ValueError(f"SMARD index empty for filter_id={filter_id} region={region}")
    parts: list[pd.DataFrame] = []
    for j, ts in enumerate(reversed(ts_list)):
        if j >= max_index_chunks:
            break
        part = fetch_smard_archive_chunk(
            base_url, filter_id, region, resolution, timestamp=str(ts)
        )
        if part.empty:
            continue
        parts.append(part)
        time.sleep(request_pause_s)
        combined = pd.concat(reversed(parts)).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        if len(combined) >= min_hours:
            break
    if not parts:
        raise ValueError(f"No SMARD rows for filter_id={filter_id}")
    combined = pd.concat(reversed(parts)).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    n = len(combined)
    if n < min_hours:
        logger.warning(
            "SMARD filter %s: stitched %s hours (target min_hours=%s, index_entries=%s, chunks_used=%s)",
            filter_id,
            n,
            min_hours,
            len(ts_list),
            len(parts),
        )
    else:
        logger.info(
            "SMARD filter %s: stitched %s hours from %s archive chunks (index_entries=%s)",
            filter_id,
            n,
            len(parts),
            len(ts_list),
        )
    return combined


def _with_retries(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 1.5) -> T:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except (requests.RequestException, ValueError) as e:
            last = e
            if i == attempts - 1:
                raise
            delay = base_delay * (2**i)
            logger.warning("request failed attempt %s/%s: %s; retry in %.1fs", i + 1, attempts, e, delay)
            time.sleep(delay)
    raise RuntimeError(last)


def _rename_single_value_column(df: pd.DataFrame, new_name: str) -> pd.DataFrame:
    if df.shape[1] != 1:
        return df
    return df.rename(columns={df.columns[0]: new_name})


def _fetch_series(dv: DataVendorConfig, filter_id: str) -> pd.DataFrame:
    def _go() -> pd.DataFrame:
        return fetch_smard_series_stitched(
            dv.base_url,
            filter_id,
            dv.region,
            dv.resolution,
            min_hours=dv.smard_min_history_hours,
            max_index_chunks=dv.smard_max_index_chunks,
        )

    return _with_retries(_go)


def _wind_sum_mw(
    dv: DataVendorConfig,
    sources: list[dict[str, str]],
    value_column: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for src in sources:
        fid = str(src["filter_id"])
        raw = _fetch_series(dv, fid)
        if raw.shape[1] != 1:
            raise ValueError(f"wind part {fid}: expected one column, got {raw.shape}")
        frames.append(raw.rename(columns={raw.columns[0]: fid}))
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="inner")
    if merged.shape[1] != len(sources):
        raise ValueError("wind join: column count mismatch")
    return merged.sum(axis=1).to_frame(value_column)


def _actual_generation_total_mw(dv: DataVendorConfig) -> pd.DataFrame:
    column = dv.table_value_columns["actual_generation_total_mw"]
    joined: pd.DataFrame | None = None
    for filter_id in dv.realized_generation_carrier_filters:
        raw = _fetch_series(dv, filter_id)
        if raw.shape[1] != 1:
            raise ValueError(f"generation {filter_id}: expected one column, got {raw.shape}")
        piece = raw.rename(columns={raw.columns[0]: filter_id})
        joined = piece if joined is None else joined.join(piece, how="outer")
    if joined is None or joined.empty:
        raise ValueError("actual generation: no carrier data")
    return joined.fillna(0.0).sum(axis=1).to_frame(column).sort_index()


def fetch_smard_bundle(config: PipelineConfig) -> DataPayload:
    dv = config.data_vendor
    if dv.type != "smard":
        raise ValueError(f"unsupported data_vendor.type: {dv.type!r}")

    series_by_name: dict[str, pd.DataFrame] = {}
    for table_name, spec in dv.bundle_single_series.items():
        fid = str(spec["filter_id"])
        series_by_name[table_name] = _rename_single_value_column(
            _fetch_series(dv, fid),
            dv.table_value_columns[table_name],
        )

    wind_forecast_mw = _wind_sum_mw(dv, dv.wind_forecast_sources, "wind_forecast_mw")
    wind_generation_actual_mw = _wind_sum_mw(
        dv,
        dv.wind_generation_sources,
        dv.table_value_columns["wind_generation_actual_mw"],
    )
    solar_generation_actual_mw = _rename_single_value_column(
        _fetch_series(dv, dv.solar_generation_actual_filter),
        dv.table_value_columns["solar_generation_actual_mw"],
    )
    actual_generation_total_mw = _actual_generation_total_mw(dv)
    hydro_pumped_storage_generation_mw = _rename_single_value_column(
        _fetch_series(dv, dv.generation_pumped_storage_filter),
        dv.table_value_columns["hydro_pumped_storage_generation_mw"],
    )

    physical_flows: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    if dv.include_physical_flows:
        on_err: Literal["skip", "raise"] = (
            "raise" if dv.on_physical_flow_error == "raise" else "skip"
        )
        for forecast_id in dv.default_physical_flow_ids:
            try:
                raw = _with_retries(
                    lambda fid=forecast_id: fetch_smard_archive_chunk(
                        dv.base_url,
                        fid,
                        dv.region,
                        dv.physical_flow_resolution,
                    )
                )
                physical_flows[forecast_id] = _rename_single_value_column(
                    raw, dv.physical_flow_value_column
                )
                logger.info("Fetched physical flow %s", forecast_id)
            except Exception as e:
                msg = str(e)
                errors[forecast_id] = msg
                logger.warning("Failed physical flow %s: %s", forecast_id, msg)
                if on_err == "raise":
                    raise

    bundle = SmardData(
        region=dv.region,
        resolution=dv.resolution,
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

    out_dir = Path(config.pipeline.output_dir)
    bundle.save(out_dir, dv.meta_filename)
    return DataPayload(bundle=bundle, vendor=dv.type)
