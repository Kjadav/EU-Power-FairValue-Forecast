"""Production fit, joblib, recursive forecast (optional LightGBM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from trading_pipeline_utils.models.inference import (
    fit_production_model,
    load_production_artifact,
    predict_recursive_hours,
    save_production_artifact,
)
from trading_pipeline_utils.settings import ModelPipelineConfig
from trading_pipeline_utils.validation.forecast_validation import validate_hourly_frame


def test_reference_history_row_prefers_24h_lag() -> None:
    cfg = ModelPipelineConfig()
    c = cfg.columns
    idx = pd.date_range("2025-06-01", periods=50, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            c.price_da: rng.standard_normal(len(idx)) * 10 + 60,
            c.load_fcst: 40_000.0 + rng.standard_normal(len(idx)) * 100,
            c.wind_fcst: 8_000.0 + rng.standard_normal(len(idx)) * 50,
            c.solar_fcst: np.clip(rng.standard_normal(len(idx)) * 500 + 3000, 0, None),
            c.hydro_fcst: 2_000.0,
            c.timestamp: idx,
        },
        index=idx,
    )
    df = validate_hourly_frame(df, cfg)
    ts = idx[-1] + pd.Timedelta(hours=5)
    # Same template row logic as ``predict_recursive_hours`` (prefers t−24h, then t−7d).
    row = None
    for delta in (pd.Timedelta(hours=24), pd.Timedelta(days=7)):
        cand = ts - delta
        if cand in df.index:
            row = df.loc[cand].copy()
            break
    assert row is not None
    assert row[c.load_fcst] == df.loc[ts - pd.Timedelta(hours=24), c.load_fcst]


def test_fit_save_load_forecast_smoke(tmp_path) -> None:
    pytest.importorskip("sklearn")
    n = 650
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    cfg = ModelPipelineConfig()
    c = cfg.columns
    cfg.hyperparam_search.enabled = False
    cfg.lgbm.n_estimators = 120
    cfg.lgbm.early_stopping_rounds = 15

    df = pd.DataFrame(
        {
            c.price_da: rng.standard_normal(n) * 20 + 50,
            c.load_fcst: 40_000 + rng.standard_normal(n) * 400,
            c.wind_fcst: 8_000 + rng.standard_normal(n) * 500,
            c.solar_fcst: np.clip(rng.standard_normal(n) * 1000 + 3000, 0, None),
            c.hydro_fcst: 2_000 + rng.standard_normal(n) * 200,
            c.wind_act: 8_000 + rng.standard_normal(n) * 500,
            c.solar_act: np.clip(rng.standard_normal(n) * 1000 + 3000, 0, None),
            c.load_act: 45_000 + rng.standard_normal(n) * 300,
            c.timestamp: idx,
        },
        index=idx,
    )
    df[c.residual_load_fcst] = (
        df[c.load_fcst] - df[c.wind_fcst] - df[c.solar_fcst] - df[c.hydro_fcst]
    )
    df = validate_hourly_frame(df, cfg)

    art = fit_production_model(df, cfg)
    path = tmp_path / "m.joblib"
    save_production_artifact(art, path)
    art2 = load_production_artifact(path)
    assert art2.train_abs_price_p95 == art.train_abs_price_p95

    fc, yt, q10t, q90t = predict_recursive_hours(df, cfg, art2, n_hours=6)
    assert len(fc) == 6
    assert yt.shape == (6,) and q10t.shape == (6,)
    assert np.isfinite(fc["y_point"].values).all()
