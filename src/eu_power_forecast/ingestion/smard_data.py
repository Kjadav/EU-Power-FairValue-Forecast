from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests

from configs.configs import META_NAME, SMARD_BASE

# Core hourly tables saved as top-level Parquet (order for QA / LLM bundles)
SMARD_CORE_TABLE_NAMES: tuple[str, ...] = (
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


def _base_url() -> str:
    return os.environ.get("SMARD_BASE", SMARD_BASE).rstrip("/")


def get_smard_index(filter_id: str, region: str, resolution: str):
    url = f"{_base_url()}/{filter_id}/{region}/index_{resolution}.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def _index_timestamp_list(raw: object) -> list:
    if isinstance(raw, dict) and "timestamps" in raw:
        return list(raw["timestamps"])
    if isinstance(raw, list):
        return raw
    raise TypeError("SMARD index JSON must be a list or {\"timestamps\": [...]}")


def get_smard_index_data(
    filter_id: str,
    region: str,
    resolution: str,
    timestamp: str | None = None,
):
    if timestamp is None:
        idx_raw = get_smard_index(filter_id, region, resolution)
        ts_list = _index_timestamp_list(idx_raw)
        if not ts_list:
            raise ValueError("SMARD index is empty; cannot resolve latest timestamp")
        timestamp = str(ts_list[-1])

    url = f"{_base_url()}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp}.json"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("data") or data.get("series") or []
    df = pd.DataFrame(rows, columns=["unix_ts", "value"])
    df["timestamp"] = pd.to_datetime(df["unix_ts"], unit="ms", utc=True)
    df = df.set_index("timestamp").drop(columns=["unix_ts"])
    return df.rename(columns={"value": f"{filter_id}_value"})


def _read_parquet_or_empty(input_dir: Path, stem: str) -> pd.DataFrame:
    path = input_dir / f"{stem}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    return pd.DataFrame()


class SmardData:
    """Standardised SMARD DE-LU bundle: fetch, save Parquet, load for training."""

    def __init__(
        self,
        region: str,
        resolution: str,
        day_ahead_prices: pd.DataFrame,
        load_forecast: pd.DataFrame,
        wind_forecast_mw: pd.DataFrame,
        solar_forecast: pd.DataFrame,
        hydro_forecast: pd.DataFrame,
        wind_generation_actual_mw: pd.DataFrame,
        solar_generation_actual_mw: pd.DataFrame,
        actual_generation_total_mw: pd.DataFrame,
        hydro_pumped_storage_generation_mw: pd.DataFrame,
        physical_flows: dict[str, pd.DataFrame] | None = None,
        physical_flow_errors: dict[str, str] | None = None,
    ) -> None:
        self.region = region
        self.resolution = resolution
        self.day_ahead_prices = day_ahead_prices
        self.load_forecast = load_forecast
        self.wind_forecast_mw = wind_forecast_mw
        self.solar_forecast = solar_forecast
        self.hydro_forecast = hydro_forecast
        self.wind_generation_actual_mw = wind_generation_actual_mw
        self.solar_generation_actual_mw = solar_generation_actual_mw
        self.actual_generation_total_mw = actual_generation_total_mw
        self.hydro_pumped_storage_generation_mw = hydro_pumped_storage_generation_mw
        self.physical_flows = dict(physical_flows) if physical_flows is not None else {}
        self.physical_flow_errors = dict(physical_flow_errors) if physical_flow_errors is not None else {}

    def iter_core_tables(self) -> tuple[tuple[str, pd.DataFrame], ...]:
        """Name and frame for each top-level bundle table (for QA / reporting)."""
        return tuple((name, getattr(self, name)) for name in SMARD_CORE_TABLE_NAMES)

    @classmethod
    def load(cls, input_dir: Path) -> SmardData:
        input_dir = Path(input_dir)
        meta_path = input_dir / META_NAME
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        region = meta.get("region", "DE-LU")
        resolution = meta.get("resolution", "hour")
        flows_dir = input_dir / "physical_flows"
        physical_flows: dict[str, pd.DataFrame] = {}
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
            hydro_forecast=_read_parquet_or_empty(input_dir, "hydro_forecast"),
            wind_generation_actual_mw=_read_parquet_or_empty(input_dir, "wind_generation_actual_mw"),
            solar_generation_actual_mw=_read_parquet_or_empty(input_dir, "solar_generation_actual_mw"),
            actual_generation_total_mw=_read_parquet_or_empty(input_dir, "actual_generation_total_mw"),
            hydro_pumped_storage_generation_mw=_read_parquet_or_empty(
                input_dir, "hydro_pumped_storage_generation_mw"
            ),
            physical_flows=physical_flows,
            physical_flow_errors=dict(meta.get("physical_flow_errors", {})),
        )

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.day_ahead_prices.to_parquet(output_dir / "day_ahead_prices.parquet")
        self.load_forecast.to_parquet(output_dir / "load_forecast.parquet")
        self.wind_forecast_mw.to_parquet(output_dir / "wind_forecast_mw.parquet")
        self.solar_forecast.to_parquet(output_dir / "solar_forecast.parquet")
        self.hydro_forecast.to_parquet(output_dir / "hydro_forecast.parquet")
        self.wind_generation_actual_mw.to_parquet(output_dir / "wind_generation_actual_mw.parquet")
        self.solar_generation_actual_mw.to_parquet(output_dir / "solar_generation_actual_mw.parquet")
        self.actual_generation_total_mw.to_parquet(output_dir / "actual_generation_total_mw.parquet")
        self.hydro_pumped_storage_generation_mw.to_parquet(
            output_dir / "hydro_pumped_storage_generation_mw.parquet"
        )
        flows_directory = output_dir / "physical_flows"
        flows_directory.mkdir(exist_ok=True)
        for fid, df in self.physical_flows.items():
            df.to_parquet(flows_directory / f"{fid}.parquet")
        meta = {
            "region": self.region,
            "resolution": self.resolution,
            "physical_flow_ids": list(self.physical_flows.keys()),
            "physical_flow_errors": self.physical_flow_errors,
            "core_tables": list(SMARD_CORE_TABLE_NAMES),
        }
        (output_dir / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
