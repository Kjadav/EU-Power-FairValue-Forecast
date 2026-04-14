import pandas as pd
import pytest

from trading_pipeline_utils.data.schemas import DataPayload, SmardData, ValidationResult
from trading_pipeline_utils.llm.data_qa import validate as llm_validate
from trading_pipeline_utils.settings import parse_pipeline_config
from trading_pipeline_utils.validation.checks import build_dataframe_profile


def test_dataframe_profile() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({"x": [1.0, float("nan"), 3.0, 4.0, 5.0]}, index=idx)
    b = build_dataframe_profile(df, "t")
    assert b["n_rows"] == 5
    assert b["missingness"]["x"] == pytest.approx(0.2)
    assert b["time_coverage"]["median_step_seconds"] == 3600.0


def test_llm_validate_skips_without_api_key() -> None:
    raw = {
        "data_vendor": {
            "type": "smard",
            "base_url": "https://example.invalid",
            "region": "DE-LU",
            "resolution": "hour",
            "physical_flow_resolution": "hour",
            "include_physical_flows": False,
            "on_physical_flow_error": "skip",
            "meta_filename": "_m.json",
            "bundle_single_series": {},
            "wind_forecast_sources": [],
            "wind_generation_sources": [],
            "solar_generation_actual_filter": "x",
            "realized_generation_carrier_filters": [],
            "generation_pumped_storage_filter": "x",
            "table_value_columns": {},
            "physical_flow_value_column": "v",
            "default_physical_flow_ids": [],
        },
        "llm": {"model": "gemini-2.5-flash"},
        "pipeline": {"output_dir": "/tmp"},
    }
    cfg = parse_pipeline_config(raw)
    empty = pd.DataFrame()
    bundle = SmardData(
        region="DE-LU",
        resolution="hour",
        day_ahead_prices=empty,
        load_forecast=empty,
        wind_forecast_mw=empty,
        solar_forecast=empty,
        hydro_forecast=empty,
        wind_generation_actual_mw=empty,
        solar_generation_actual_mw=empty,
        actual_generation_total_mw=empty,
        hydro_pumped_storage_generation_mw=empty,
    )
    payload = DataPayload(bundle=bundle, vendor="smard")
    vr = ValidationResult(ok=True, tables={}, summary={})
    out = llm_validate(payload, vr, cfg)
    assert out.verdict == "review"
    assert out.confidence == 0.0
    assert any(i.get("code") == "llm_skipped" for i in out.issues)
