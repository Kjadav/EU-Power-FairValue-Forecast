from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from trading_pipeline_utils.features.engineering import compute_residual_load
from trading_pipeline_utils.settings import ModelPipelineConfig
from trading_pipeline_utils.validation.forecast_validation import validate_hourly_frame

logger = logging.getLogger(__name__)

def load_smard_bundle_hourly(bundle_dir: Path, config: ModelPipelineConfig) -> pd.DataFrame:
    d = Path(bundle_dir)
    c = config.columns

    def one_col(stem: str, canon: str) -> pd.Series:
        p = d / f"{stem}.parquet"
        if not p.is_file():
            raise FileNotFoundError(p)
        df = pd.read_parquet(p)
        if df.shape[1] != 1:
            raise ValueError(f"{stem}: expected 1 column, got {df.shape[1]}")
        return df.iloc[:, 0].rename(canon)

    pieces: dict[str, pd.Series] = {
        c.price_da: one_col("day_ahead_prices", c.price_da),
        c.load_fcst: one_col("load_forecast", c.load_fcst),
        c.wind_fcst: one_col("wind_forecast_mw", c.wind_fcst),
        c.solar_fcst: one_col("solar_forecast", c.solar_fcst),
        c.hydro_fcst: one_col("hydro_forecast", c.hydro_fcst),
    }

    for stem, name in (
        ("wind_generation_actual_mw", c.wind_act),
        ("solar_generation_actual_mw", c.solar_act),
    ):
        p = d / f"{stem}.parquet"
        if p.is_file():
            df = pd.read_parquet(p)
            if df.shape[1] == 1:
                pieces[name] = df.iloc[:, 0].rename(name)

    gen_p = d / "actual_generation_total_mw.parquet"
    if gen_p.is_file():
        gdf = pd.read_parquet(gen_p)
        if gdf.shape[1] == 1:
            pieces[c.load_act] = gdf.iloc[:, 0].rename(c.load_act)
            logger.info("Mapped actual_generation_total_mw -> %s", c.load_act)

    hydro_act_p = d / "hydro_generation_actual_mw.parquet"
    if hydro_act_p.is_file():
        hdf = pd.read_parquet(hydro_act_p)
        if hdf.shape[1] == 1:
            pieces[c.hydro_act] = hdf.iloc[:, 0].rename(c.hydro_act)

    out = pd.DataFrame(pieces).sort_index()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out[~out.index.duplicated(keep="first")]
    out[c.residual_load_fcst] = compute_residual_load(out, c)

    out[c.timestamp] = out.index
    return validate_hourly_frame(out, config)
