from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

META_NAME = "_smard_bundle_meta.json"

_FILTER_DAY_AHEAD_PRICES = "4169"
_FILTER_LOAD_FORECAST = "410"
_FILTER_WIND_FORECAST_ONSHORE = "123"
_FILTER_WIND_FORECAST_OFFSHORE = "3791"
_FILTER_SOLAR_FORECAST = "125"
_FILTER_HYDRO_FORECAST = "715"
_FILTER_WIND_GENERATION_ONSHORE = "4067"
_FILTER_WIND_GENERATION_OFFSHORE = "1225"
_FILTER_SOLAR_GENERATION_ACTUAL = "4068"
_FILTER_GENERATION_PUMPED_STORAGE = "4070"

_REALIZED_GENERATION_CARRIER_FILTERS = (
    "1223", "1224", _FILTER_WIND_GENERATION_OFFSHORE, "1226", "1227",
    "1228", "4066", _FILTER_WIND_GENERATION_ONSHORE,
    _FILTER_SOLAR_GENERATION_ACTUAL, "4069", "4070", "4071",
)

_SINGLE_SERIES: dict[str, tuple[str, str]] = {
    "day_ahead_prices": (_FILTER_DAY_AHEAD_PRICES, "day_ahead_price_eur_mwh"),
    "load_forecast": (_FILTER_LOAD_FORECAST, "total_load_mw"),
    "solar_forecast": (_FILTER_SOLAR_FORECAST, "solar_forecast_mw"),
    "hydro_forecast": (_FILTER_HYDRO_FORECAST, "hydro_forecast_mw"),
}

_WIND_FORECAST_SOURCES = (
    (_FILTER_WIND_FORECAST_ONSHORE, "onshore"),
    (_FILTER_WIND_FORECAST_OFFSHORE, "offshore"),
)

_WIND_GENERATION_SOURCES = (
    (_FILTER_WIND_GENERATION_ONSHORE, "onshore"),
    (_FILTER_WIND_GENERATION_OFFSHORE, "offshore"),
)

_VALUE_COLUMNS: dict[str, str] = {
    "wind_generation_actual_mw": "wind_generation_actual_mw",
    "solar_generation_actual_mw": "solar_generation_actual_mw",
    "actual_generation_total_mw": "actual_generation_total_mw",
    "hydro_pumped_storage_generation_mw": "hydro_pumped_storage_generation_mw",
}

CORE_TABLE_NAMES: tuple[str, ...] = (
    "day_ahead_prices",
    "load_forecast",
    "wind_forecast_mw",
    "solar_forecast",
    "hydro_forecast",
    "wind_generation_actual_mw",
    "solar_generation_actual_mw",
    "actual_generation_total_mw",
    "hydro_pumped_storage_generation_mw",
)

_DEFAULT_PHYSICAL_FLOW_IDS = (
    "22004629", "22004406", "22004548", "22004410", "22004552",
    "22004403", "22004545", "22004412", "22004553", "22004405",
)


@dataclass
class DataPayload:
    region: str
    resolution: str
    day_ahead_prices: pd.DataFrame
    load_forecast: pd.DataFrame
    wind_forecast_mw: pd.DataFrame
    solar_forecast: pd.DataFrame
    hydro_forecast: pd.DataFrame
    wind_generation_actual_mw: pd.DataFrame
    solar_generation_actual_mw: pd.DataFrame
    actual_generation_total_mw: pd.DataFrame
    hydro_pumped_storage_generation_mw: pd.DataFrame
    physical_flows: dict[str, pd.DataFrame] = field(default_factory=dict)
    physical_flow_errors: dict[str, str] = field(default_factory=dict)

    def iter_core_tables(self) -> tuple[tuple[str, pd.DataFrame], ...]:
        return tuple((name, getattr(self, name)) for name in CORE_TABLE_NAMES)

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in self.iter_core_tables():
            df.to_parquet(output_dir / f"{name}.parquet")
        flows_dir = output_dir / "physical_flows"
        flows_dir.mkdir(exist_ok=True)
        for fid, df in self.physical_flows.items():
            df.to_parquet(flows_dir / f"{fid}.parquet")
        meta = {
            "region": self.region,
            "resolution": self.resolution,
            "physical_flow_ids": list(self.physical_flows.keys()),
            "physical_flow_errors": self.physical_flow_errors,
            "core_tables": list(CORE_TABLE_NAMES),
        }
        (output_dir / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, input_dir: Path) -> DataPayload:
        input_dir = Path(input_dir)
        meta_path = input_dir / META_NAME
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        region = meta.get("region", "DE-LU")
        resolution = meta.get("resolution", "hour")

        def _read(stem: str) -> pd.DataFrame:
            p = input_dir / f"{stem}.parquet"
            return pd.read_parquet(p) if p.is_file() else pd.DataFrame()

        physical_flows: dict[str, pd.DataFrame] = {}
        flows_dir = input_dir / "physical_flows"
        if flows_dir.is_dir():
            for p in sorted(flows_dir.glob("*.parquet")):
                physical_flows[p.stem] = pd.read_parquet(p)

        return cls(
            region=region,
            resolution=resolution,
            day_ahead_prices=pd.read_parquet(input_dir / "day_ahead_prices.parquet"),
            load_forecast=pd.read_parquet(input_dir / "load_forecast.parquet"),
            wind_forecast_mw=pd.read_parquet(input_dir / "wind_forecast_mw.parquet"),
            solar_forecast=pd.read_parquet(input_dir / "solar_forecast.parquet"),
            hydro_forecast=_read("hydro_forecast"),
            wind_generation_actual_mw=_read("wind_generation_actual_mw"),
            solar_generation_actual_mw=_read("solar_generation_actual_mw"),
            actual_generation_total_mw=_read("actual_generation_total_mw"),
            hydro_pumped_storage_generation_mw=_read("hydro_pumped_storage_generation_mw"),
            physical_flows=physical_flows,
            physical_flow_errors=dict(meta.get("physical_flow_errors", {})),
        )


def _get_index(base_url: str, filter_id: str, region: str, resolution: str) -> list[int]:
    url = f"{base_url}/{filter_id}/{region}/index_{resolution}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    if isinstance(raw, dict) and "timestamps" in raw:
        return list(raw["timestamps"])
    if isinstance(raw, list):
        return raw
    raise TypeError("SMARD index JSON: expected list or {timestamps: [...]}")


def _get_series(
    base_url: str, filter_id: str, region: str, resolution: str, timestamp: str | None = None
) -> pd.DataFrame:
    if timestamp is None:
        ts_list = _get_index(base_url, filter_id, region, resolution)
        if not ts_list:
            raise ValueError("SMARD index is empty")
        timestamp = str(ts_list[-1])
    url = f"{base_url}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("data") or payload.get("series") or []
    df = pd.DataFrame(rows, columns=["unix_ts", "value"])
    df["timestamp"] = pd.to_datetime(df["unix_ts"], unit="ms", utc=True)
    df = df.set_index("timestamp").drop(columns=["unix_ts"])
    return df.rename(columns={"value": f"{filter_id}_value"})


def _rename_col(df: pd.DataFrame, new_name: str) -> pd.DataFrame:
    if df.shape[1] != 1:
        return df
    return df.rename(columns={df.columns[0]: new_name})


def _wind_sum(
    base_url: str,
    region: str,
    resolution: str,
    sources: tuple[tuple[str, str], ...],
    col_name: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for filter_id, _label in sources:
        raw = _get_series(base_url, filter_id, region, resolution)
        if raw.shape[1] != 1:
            raise ValueError(f"wind {filter_id}: expected one column, got {raw.shape}")
        frames.append(raw.rename(columns={raw.columns[0]: filter_id}))
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="inner")
    return merged.sum(axis=1).to_frame(col_name)


def _total_generation(base_url: str, region: str, resolution: str) -> pd.DataFrame:
    joined: pd.DataFrame | None = None
    for filter_id in _REALIZED_GENERATION_CARRIER_FILTERS:
        raw = _get_series(base_url, filter_id, region, resolution)
        if raw.shape[1] != 1:
            raise ValueError(f"generation {filter_id}: expected one column, got {raw.shape}")
        piece = raw.rename(columns={raw.columns[0]: filter_id})
        joined = piece if joined is None else joined.join(piece, how="outer")
    if joined is None or joined.empty:
        raise ValueError("actual generation: no carrier data")
    col = _VALUE_COLUMNS["actual_generation_total_mw"]
    return joined.fillna(0.0).sum(axis=1).to_frame(col).sort_index()


def fetch(config: dict[str, Any]) -> DataPayload:
    """Fetch all SMARD DE-LU series and return a normalised DataPayload."""
    vendor = config.get("data_vendor", {})
    base_url = vendor.get("base_url", "https://www.smard.de/app/chart_data").rstrip("/")
    region = vendor.get("region", "DE-LU")
    resolution = vendor.get("resolution", "hour")

    pf_cfg = vendor.get("physical_flows", {})
    include_flows = pf_cfg.get("enabled", True)
    flow_resolution = pf_cfg.get("resolution", "quarterhour")
    flow_ids: list[str] = pf_cfg.get("ids", list(_DEFAULT_PHYSICAL_FLOW_IDS))

    series: dict[str, pd.DataFrame] = {}
    for table_name, (filter_id, col_name) in _SINGLE_SERIES.items():
        series[table_name] = _rename_col(
            _get_series(base_url, filter_id, region, resolution), col_name
        )

    wind_fc = _wind_sum(base_url, region, resolution, _WIND_FORECAST_SOURCES, "wind_forecast_mw")
    wind_gen = _wind_sum(
        base_url, region, resolution, _WIND_GENERATION_SOURCES,
        _VALUE_COLUMNS["wind_generation_actual_mw"],
    )
    solar_gen = _rename_col(
        _get_series(base_url, _FILTER_SOLAR_GENERATION_ACTUAL, region, resolution),
        _VALUE_COLUMNS["solar_generation_actual_mw"],
    )
    gen_total = _total_generation(base_url, region, resolution)
    hydro_ps = _rename_col(
        _get_series(base_url, _FILTER_GENERATION_PUMPED_STORAGE, region, resolution),
        _VALUE_COLUMNS["hydro_pumped_storage_generation_mw"],
    )

    physical_flows: dict[str, pd.DataFrame] = {}
    flow_errors: dict[str, str] = {}
    if include_flows:
        for fid in flow_ids:
            try:
                raw = _get_series(base_url, fid, region, flow_resolution)
                physical_flows[fid] = _rename_col(raw, "physical_flow_mw")
                logger.info("fetched physical flow %s", fid)
            except Exception as exc:
                flow_errors[fid] = str(exc)
                logger.warning("failed physical flow %s: %s", fid, exc)

    output_dir = vendor.get("output_dir")
    payload = DataPayload(
        region=region,
        resolution=resolution,
        day_ahead_prices=series["day_ahead_prices"],
        load_forecast=series["load_forecast"],
        wind_forecast_mw=wind_fc,
        solar_forecast=series["solar_forecast"],
        hydro_forecast=series["hydro_forecast"],
        wind_generation_actual_mw=wind_gen,
        solar_generation_actual_mw=solar_gen,
        actual_generation_total_mw=gen_total,
        hydro_pumped_storage_generation_mw=hydro_ps,
        physical_flows=physical_flows,
        physical_flow_errors=flow_errors,
    )
    if output_dir:
        payload.save(Path(output_dir))
        logger.info("saved bundle to %s", output_dir)
    return payload
