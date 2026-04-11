"""Tiny hourly SMARD-shaped bundle for tests / demos (no matplotlib)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.forecaster import PRICE_COL


def synthetic_hourly_index(n_hours: int) -> pd.DatetimeIndex:
    """Hourly UTC index ending at the current UTC hour."""
    end = pd.Timestamp.now(tz="UTC").floor("h")
    return pd.date_range(end=end, periods=n_hours, freq="h", tz="UTC")


def write_synthetic_regression_bundle(path: Path, n_hours: int = 35 * 24) -> None:
    """Minimal bundle when real SMARD slice is too short for P_{t-168}."""
    path.mkdir(parents=True, exist_ok=True)
    idx = synthetic_hourly_index(n_hours)
    rng = np.random.default_rng(0)
    load = 40_000 + rng.standard_normal(n_hours) * 500
    wind = 10_000 + rng.standard_normal(n_hours) * 800
    solar = np.clip(rng.standard_normal(n_hours) * 2_000 + 5_000, 0, None)
    hydro = 3_000 + rng.standard_normal(n_hours) * 400
    residual = load - wind - solar - hydro
    noise = rng.standard_normal(n_hours) * 8.0
    price = 50.0 + 0.002 * residual + noise
    pd.DataFrame({PRICE_COL: price}, index=idx).to_parquet(path / "day_ahead_prices.parquet")
    pd.DataFrame({"total_load_mw": load}, index=idx).to_parquet(path / "load_forecast.parquet")
    pd.DataFrame({"wind_forecast_mw": wind}, index=idx).to_parquet(path / "wind_forecast_mw.parquet")
    pd.DataFrame({"solar_forecast_mw": solar}, index=idx).to_parquet(path / "solar_forecast.parquet")
    pd.DataFrame({"hydro_forecast_mw": hydro}, index=idx).to_parquet(path / "hydro_forecast.parquet")
