
SMARD_BASE = "https://www.smard.de/app/chart_data"

# Manifest filename written next to bundle Parquet files
META_NAME = "_smard_bundle_meta.json"

# ---------------------------------------------------------------------------
# SMARD filter IDs (DE-LU, hourly unless noted) — verify on smard.de if a series moves
# ---------------------------------------------------------------------------

FILTER_DAY_AHEAD_PRICES = "4169"  # Marktpreis DE/LU (EUR/MWh)
FILTER_LOAD_FORECAST = "410"  # Stromverbrauch Gesamt (Netzlast) DE-LU

# --- Prognostizierte Erzeugung (day-ahead TSO forecast, MW) ---
FILTER_WIND_FORECAST_ONSHORE = "123"
FILTER_WIND_FORECAST_OFFSHORE = "3791"
FILTER_SOLAR_FORECAST = "125"
# "Sonstige" day-ahead generation forecast (SMARD 715): non-wind/PV share — includes run-of-river hydro,
# biomass, geothermal, etc. SMARD does not publish a separate *only-hydro* prognose; adjust if they add one.
FILTER_HYDRO_FORECAST = "715"

# --- Realisierte Erzeugung (actual / metered net generation, MW) ---
FILTER_WIND_GENERATION_ONSHORE = "4067"
FILTER_WIND_GENERATION_OFFSHORE = "1225"
FILTER_SOLAR_GENERATION_ACTUAL = "4068"

# Backward-compatible names (= actual generation filters, used across codebase)
FILTER_WIND_ONSHORE = FILTER_WIND_GENERATION_ONSHORE
FILTER_WIND_OFFSHORE = FILTER_WIND_GENERATION_OFFSHORE

# Realized generation — Pumpspeicher **Erzeugung** (net generation, MW)
FILTER_GENERATION_PUMPED_STORAGE = "4070"

# Realized net generation by carrier (SMARD “Stromerzeugung”); summed ≈ total realized generation
REALIZED_GENERATION_CARRIER_FILTERS: tuple[str, ...] = (
    "1223",  # Braunkohle
    "1224",  # Kernenergie
    FILTER_WIND_GENERATION_OFFSHORE,
    "1226",  # Wasserkraft
    "1227",  # Sonstige Konventionelle
    "1228",  # Sonstige Erneuerbare
    "4066",  # Biomasse
    FILTER_WIND_GENERATION_ONSHORE,
    FILTER_SOLAR_GENERATION_ACTUAL,
    "4069",  # Steinkohle
    "4070",  # Pumpspeicher
    "4071",  # Erdgas
)

REALIZED_GENERATION_CARRIER_LABELS: dict[str, str] = {
    "1223": "lignite",
    "1224": "nuclear",
    "1225": "wind_offshore",
    "1226": "hydro_run_of_river",
    "1227": "other_conventional",
    "1228": "other_renewable",
    "4066": "biomass",
    "4067": "wind_onshore",
    "4068": "solar_pv",
    "4069": "hard_coal",
    "4070": "pumped_storage_generation",
    "4071": "natural_gas",
}

# Single-series bundle tables: parquet stem → (SMARD filter_id, short description)
SMARD_BUNDLE_SINGLE_SERIES: dict[str, tuple[str, str]] = {
    "day_ahead_prices": (FILTER_DAY_AHEAD_PRICES, "Day-ahead price DE-LU"),
    "load_forecast": (FILTER_LOAD_FORECAST, "Total load forecast DE-LU"),
    "solar_forecast": (FILTER_SOLAR_FORECAST, "Solar PV day-ahead forecast DE-LU (prognose)"),
    "hydro_forecast": (
        FILTER_HYDRO_FORECAST,
        "Hydro-heavy 'other' generation day-ahead forecast DE-LU (SMARD Sonstige prognose)",
    ),
}

# Wind **forecast** = onshore + offshore prognose filters
SMARD_WIND_FORECAST_SOURCES: tuple[tuple[str, str], ...] = (
    (FILTER_WIND_FORECAST_ONSHORE, "onshore"),
    (FILTER_WIND_FORECAST_OFFSHORE, "offshore"),
)

# Wind **actual generation** = onshore + offshore realized filters
SMARD_WIND_GENERATION_SOURCES: tuple[tuple[str, str], ...] = (
    (FILTER_WIND_GENERATION_ONSHORE, "onshore"),
    (FILTER_WIND_GENERATION_OFFSHORE, "offshore"),
)

# Canonical value column names after ingest (replaces ``{filter_id}_value`` where applicable)
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

PHYSICAL_FLOW_VALUE_COLUMN = "physical_flow_mw"

DEFAULT_PHYSICAL_FLOW_IDS: tuple[str, ...] = (
    "22004629",
    "22004406",
    "22004548",
    "22004410",
    "22004552",
    "22004403",
    "22004545",
    "22004412",
    "22004553",
    "22004405",
)
