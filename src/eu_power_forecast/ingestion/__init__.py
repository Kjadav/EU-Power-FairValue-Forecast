from configs.configs import (
    DEFAULT_PHYSICAL_FLOW_IDS,
    FILTER_DAY_AHEAD_PRICES,
    FILTER_GENERATION_PUMPED_STORAGE,
    FILTER_HYDRO_FORECAST,
    FILTER_LOAD_FORECAST,
    FILTER_SOLAR_FORECAST,
    FILTER_SOLAR_GENERATION_ACTUAL,
    FILTER_WIND_FORECAST_OFFSHORE,
    FILTER_WIND_FORECAST_ONSHORE,
    FILTER_WIND_GENERATION_OFFSHORE,
    FILTER_WIND_GENERATION_ONSHORE,
    FILTER_WIND_OFFSHORE,
    FILTER_WIND_ONSHORE,
    META_NAME,
    REALIZED_GENERATION_CARRIER_FILTERS,
)
from eu_power_forecast.ingestion.fetch_smard_data import download_smard_data_de_lu
from eu_power_forecast.ingestion.smard_data import SMARD_CORE_TABLE_NAMES, SmardData
from eu_power_forecast.utils.quality import execute_llm_rules, parse_llm_rules_response

_LLM_QA_EXPORTS = frozenset(
    {"baseline_qa_dataframe", "dataframe_profile", "qa_smard_bundle_report", "qa_single_table"}
)


def __getattr__(name: str):
    if name in _LLM_QA_EXPORTS:
        from eu_power_forecast.llm_integration.data_fetch.quality import llm_data_qa as _llm_qa

        return getattr(_llm_qa, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_PHYSICAL_FLOW_IDS",
    "FILTER_DAY_AHEAD_PRICES",
    "FILTER_GENERATION_PUMPED_STORAGE",
    "FILTER_HYDRO_FORECAST",
    "FILTER_LOAD_FORECAST",
    "FILTER_SOLAR_FORECAST",
    "FILTER_SOLAR_GENERATION_ACTUAL",
    "FILTER_WIND_FORECAST_OFFSHORE",
    "FILTER_WIND_FORECAST_ONSHORE",
    "FILTER_WIND_GENERATION_OFFSHORE",
    "FILTER_WIND_GENERATION_ONSHORE",
    "FILTER_WIND_OFFSHORE",
    "FILTER_WIND_ONSHORE",
    "META_NAME",
    "REALIZED_GENERATION_CARRIER_FILTERS",
    "SMARD_CORE_TABLE_NAMES",
    "SmardData",
    "baseline_qa_dataframe",
    "dataframe_profile",
    "download_smard_data_de_lu",
    "execute_llm_rules",
    "parse_llm_rules_response",
    "qa_single_table",
    "qa_smard_bundle_report",
]
