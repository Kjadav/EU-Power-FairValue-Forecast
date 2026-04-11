"""Business meaning for SMARD tables/columns — passed to the LLM alongside samples."""

from __future__ import annotations

from configs.configs import (
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
    REALIZED_GENERATION_CARRIER_FILTERS,
)

TABLE_BUSINESS_CONTEXT: dict[str, str] = {
    "day_ahead_prices": (
        "Hourly **day-ahead** wholesale electricity price for bidding zone **DE-LU** (Germany–Luxembourg). "
        "Values are **EUR/MWh**. Negative prices are normal when renewables oversupply the grid."
    ),
    "load_forecast": (
        "**Total load / net consumption** for DE-LU (SMARD series tied to filter "
        f"{FILTER_LOAD_FORECAST}). Typically **MW**; should be strictly positive at scale."
    ),
    "wind_forecast_mw": (
        "**Day-ahead forecast** of combined on- and offshore **wind generation** in DE-LU (MW). "
        f"SMARD *prognostizierte Erzeugung*: filters {FILTER_WIND_FORECAST_ONSHORE} (onshore) + "
        f"{FILTER_WIND_FORECAST_OFFSHORE} (offshore)."
    ),
    "solar_forecast": (
        "**Day-ahead forecast** of **solar PV generation** for DE-LU (MW). "
        f"SMARD *prognostizierte Erzeugung*, filter {FILTER_SOLAR_FORECAST}."
    ),
    "hydro_forecast": (
        "**Day-ahead forecast** for SMARD **„Sonstige“** generation (MW): mainly run-of-river **hydro**, "
        "biomass, geothermal, and other non-wind/PV renewables — not wind/PV themselves. "
        f"Filter {FILTER_HYDRO_FORECAST} (*Prognostizierte Erzeugung: Sonstige*)."
    ),
    "wind_generation_actual_mw": (
        "**Realized (metered) net wind generation** for DE-LU (MW): on- + offshore. "
        f"SMARD *realisierte Erzeugung*: {FILTER_WIND_GENERATION_ONSHORE} + "
        f"{FILTER_WIND_GENERATION_OFFSHORE}."
    ),
    "solar_generation_actual_mw": (
        "**Realized (metered) solar PV generation** for DE-LU (MW). "
        f"SMARD *realisierte Erzeugung*, filter {FILTER_SOLAR_GENERATION_ACTUAL}."
    ),
    "actual_generation_total_mw": (
        "**Realized (actual) net generation** for DE-LU (**MW**): sum of SMARD realized-generation "
        f"carriers {', '.join(REALIZED_GENERATION_CARRIER_FILTERS)} (≈ total net generation by fuel mix)."
    ),
    "hydro_pumped_storage_generation_mw": (
        "**Pumped-storage hydro generation** (net generation into the grid, **MW**). "
        f"SMARD filter {FILTER_GENERATION_PUMPED_STORAGE}."
    ),
}


def _column_matches_filter_suffix(column: str, filter_id: str) -> bool:
    return column == f"{filter_id}_value" or column.startswith(f"{filter_id}_")


CANONICAL_COLUMN_DESCRIPTIONS: dict[str, str] = {
    "day_ahead_price_eur_mwh": (
        "DE-LU **day-ahead** wholesale electricity price (**EUR/MWh**); negative values are possible."
    ),
    "total_load_mw": (
        f"**Total load / net consumption** for DE-LU (**MW**), SMARD filter {FILTER_LOAD_FORECAST}."
    ),
    "solar_forecast_mw": (
        f"**Solar PV day-ahead forecast** (**MW**), SMARD prognose filter {FILTER_SOLAR_FORECAST}."
    ),
    "hydro_forecast_mw": (
        f"**Sonstige** day-ahead generation forecast (**MW**), SMARD filter {FILTER_HYDRO_FORECAST} "
        "(hydro-rich aggregate, not wind/PV)."
    ),
    "wind_forecast_mw": (
        f"Combined on- + offshore **wind day-ahead forecast** (**MW**); prognose filters "
        f"{FILTER_WIND_FORECAST_ONSHORE} + {FILTER_WIND_FORECAST_OFFSHORE}."
    ),
    "wind_generation_actual_mw": (
        f"**Realized wind generation** on- + offshore (**MW**); filters "
        f"{FILTER_WIND_GENERATION_ONSHORE} + {FILTER_WIND_GENERATION_OFFSHORE}."
    ),
    "solar_generation_actual_mw": (
        f"**Realized solar PV generation** (**MW**), filter {FILTER_SOLAR_GENERATION_ACTUAL}."
    ),
    "actual_generation_total_mw": (
        "Sum of realized net generation by carrier for DE-LU (**MW**); built from SMARD filters "
        f"{', '.join(REALIZED_GENERATION_CARRIER_FILTERS)}."
    ),
    "hydro_pumped_storage_generation_mw": (
        f"**Pumped-storage generation** (**MW**), SMARD filter {FILTER_GENERATION_PUMPED_STORAGE}."
    ),
    "physical_flow_mw": (
        "**Cross-border physical flow** (or related SMARD flow series) in **MW**; sign convention per SMARD."
    ),
}


def describe_column(table_name: str, column: str) -> str | None:
    """One-line business description for a dataframe column, if known."""
    if column in CANONICAL_COLUMN_DESCRIPTIONS:
        return CANONICAL_COLUMN_DESCRIPTIONS[column]

    if _column_matches_filter_suffix(column, FILTER_DAY_AHEAD_PRICES):
        return (
            f"DE-LU day-ahead market clearing price (EUR/MWh), SMARD filter {FILTER_DAY_AHEAD_PRICES}. "
            "May be negative."
        )
    if _column_matches_filter_suffix(column, FILTER_LOAD_FORECAST):
        return (
            f"System load / net demand proxy (MW), SMARD filter {FILTER_LOAD_FORECAST}. "
            "Expect large positive values."
        )
    if _column_matches_filter_suffix(column, FILTER_SOLAR_FORECAST):
        return f"Solar PV forecast (MW), SMARD prognose filter {FILTER_SOLAR_FORECAST}."
    if _column_matches_filter_suffix(column, FILTER_HYDRO_FORECAST):
        return f"Hydro-rich 'Sonstige' generation forecast (MW), filter {FILTER_HYDRO_FORECAST}."
    if _column_matches_filter_suffix(column, FILTER_WIND_FORECAST_ONSHORE):
        return f"Onshore wind forecast (MW), SMARD prognose filter {FILTER_WIND_FORECAST_ONSHORE}."
    if _column_matches_filter_suffix(column, FILTER_WIND_FORECAST_OFFSHORE):
        return f"Offshore wind forecast (MW), SMARD prognose filter {FILTER_WIND_FORECAST_OFFSHORE}."
    if _column_matches_filter_suffix(column, FILTER_WIND_GENERATION_ONSHORE):
        return f"Onshore wind actual generation (MW), filter {FILTER_WIND_GENERATION_ONSHORE}."
    if _column_matches_filter_suffix(column, FILTER_WIND_GENERATION_OFFSHORE):
        return f"Offshore wind actual generation (MW), filter {FILTER_WIND_GENERATION_OFFSHORE}."
    if _column_matches_filter_suffix(column, FILTER_SOLAR_GENERATION_ACTUAL):
        return f"Solar PV actual generation (MW), filter {FILTER_SOLAR_GENERATION_ACTUAL}."

    if table_name == "day_ahead_prices" and column.endswith("_value"):
        return "Price or indexed monetary value in EUR/MWh (verify filter id in column name)."
    if table_name == "load_forecast" and column.endswith("_value"):
        return "Load-related numeric series (typically MW)."
    if table_name == "solar_forecast" and column.endswith("_value"):
        return "Solar forecast (MW)."
    return None


def business_context_block(table_name: str, columns: list[str]) -> str:
    """Markdown-style block for the LLM user message."""
    lines: list[str] = ["## Business context", ""]
    table_blurb = TABLE_BUSINESS_CONTEXT.get(table_name)
    if table_blurb:
        lines.append(f"**Table `{table_name}`:** {table_blurb}")
        lines.append("")
    lines.append("**Columns:**")
    for column in columns:
        description = describe_column(table_name, str(column))
        if description:
            lines.append(f"- `{column}`: {description}")
        else:
            lines.append(f"- `{column}`: (no curated description — infer cautiously from dtype and samples)")
    return "\n".join(lines)
