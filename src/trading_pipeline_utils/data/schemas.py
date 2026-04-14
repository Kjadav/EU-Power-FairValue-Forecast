"""
Types for SMARD table, validation function and LLM functions
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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

SMARD_TABLE_VALUE_COLUMNS: dict[str, str] = {
    "day_ahead_prices": "day_ahead_price_eur_mwh",
    "load_forecast": "total_load_mw",
    "solar_forecast": "solar_forecast_mw",
    "hydro_forecast": "hydro_forecast_mw",
    "wind_generation_actual_mw": "wind_generation_actual_mw",
    "solar_generation_actual_mw": "solar_generation_actual_mw",
    "actual_generation_total_mw": "actual_generation_total_mw",
    "hydro_pumped_storage_generation_mw": "hydro_pumped_storage_generation_mw",
}

TABLE_BUSINESS_CONTEXT: dict[str, str] = {
    "day_ahead_prices": (
        "Hourly day-ahead wholesale electricity price for bidding zone DE-LU (EUR/MWh). "
        "Negative prices are possible."
    ),
    "load_forecast": "Total load / net consumption for DE-LU (MW); typically strictly positive at scale.",
    "wind_forecast_mw": "Day-ahead forecast of combined on- and offshore wind generation in DE-LU (MW).",
    "solar_forecast": "Day-ahead forecast of solar PV generation for DE-LU (MW).",
    "hydro_forecast": (
        "Day-ahead 'Sonstige' generation forecast (MW): hydro-heavy aggregate and other non-wind/PV."
    ),
    "wind_generation_actual_mw": "Realized net wind generation for DE-LU (MW), on- plus offshore.",
    "solar_generation_actual_mw": "Realized solar PV generation for DE-LU (MW).",
    "actual_generation_total_mw": "Sum of realized net generation by carrier for DE-LU (MW).",
    "hydro_pumped_storage_generation_mw": "Pumped-storage hydro net generation into the grid (MW).",
}


def _read_parquet_or_empty(input_dir: Path, stem: str) -> pd.DataFrame:
    path = input_dir / f"{stem}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    return pd.DataFrame()


@dataclass
class SmardData:
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
        return ((name, getattr(self, name)) for name in SMARD_CORE_TABLE_NAMES)

    @classmethod
    def load(cls, input_dir: Path, meta_filename: str) -> SmardData:
        input_dir = Path(input_dir)
        meta_path = input_dir / meta_filename
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

    def save(self, output_dir: Path, meta_filename: str) -> None:
        """
        the type for the template to save to the results directory
        """
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
        (output_dir / meta_filename).write_text(json.dumps(meta, indent=2), encoding="utf-8")


@dataclass
class ValidationResult:
    """
    success type
    
    ok : bool - for success/failure
    tables: dict - table
    summary: dict - tables with details
    """

    ok: bool
    tables: dict[str, dict[str, object]]
    summary: dict[str, object]


@dataclass
class DataPayload:
    """
    bundle: SmardData - data that is fetched from smard.de
    vendor - smard.de
    fetched_at_utc - time of run
    validation - validation result
    """

    bundle: SmardData
    vendor: str
    fetched_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    validation: ValidationResult | None = None


def build_table_context_markdown(table_name: str, columns: list[str]) -> str:
    lines: list[str] = ["## Business context", ""]
    blurb = TABLE_BUSINESS_CONTEXT.get(table_name)
    if blurb:
        lines.append(f"**Table `{table_name}`:** {blurb}")
        lines.append("")
    lines.append("**Columns:**")
    for column in columns:
        lines.append(f"- `{column}`")
    return "\n".join(lines)
