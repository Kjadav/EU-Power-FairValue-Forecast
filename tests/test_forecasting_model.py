"""Target transform, quantile ordering, causality, weekly helpers, deterministic LightGBM."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_pipeline_utils.features.engineering import build_feature_matrix
from trading_pipeline_utils.features.transforms import ArcsinhPriceTransform
from trading_pipeline_utils.forecasting.leakage import assert_truncated_history_matches_tail
from trading_pipeline_utils.forecasting.metrics import enforce_quantile_monotonicity, weekly_baseload_peakload
from trading_pipeline_utils.forecasting.simulation import encode_bucket
from trading_pipeline_utils.settings import ModelPipelineConfig
from trading_pipeline_utils.validation.forecast_validation import validate_hourly_frame


def test_arcsinh_inverse_negative_and_zero() -> None:
    t = ArcsinhPriceTransform(scale_floor=10.0)
    t.fit(np.array([-300.0, -50.0, 0.0, 40.0, 120.0]))
    y = np.array([-80.0, 0.0, 25.0, -0.01])
    z = t.transform(y)
    back = t.inverse_transform(z)
    np.testing.assert_allclose(back, y, rtol=1e-6, atol=1e-4)


def test_enforce_quantile_monotonicity() -> None:
    q10 = np.array([3.0, 1.0])
    q50 = np.array([2.0, 2.0])
    q90 = np.array([1.0, 4.0])
    a, b, c = enforce_quantile_monotonicity(q10, q50, q90)
    assert np.all(a <= b) and np.all(b <= c)


def test_weekly_baseload_peakload_mask() -> None:
    h = np.arange(24 * 7, dtype=float)
    peak = np.zeros(len(h), dtype=bool)
    peak[8:20] = True
    base, peak_ld = weekly_baseload_peakload(h, peak)
    assert base == pytest.approx(np.mean(h))
    assert peak_ld == pytest.approx(np.mean(h[peak]))


def test_encode_bucket_deterministic() -> None:
    b = encode_bucket(
        np.array([10, 10]),
        np.array([0, 1]),
        np.array([3, 3]),
    )
    assert b[0] != b[1]


def test_feature_causality_numeric() -> None:
    n = 400
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    cfg = ModelPipelineConfig()
    c = cfg.columns
    df = pd.DataFrame(
        {
            c.price_da: rng.standard_normal(n) * 20 + 50,
            c.load_fcst: 40_000 + rng.standard_normal(n) * 400,
            c.wind_fcst: 8_000 + rng.standard_normal(n) * 500,
            c.solar_fcst: np.clip(rng.standard_normal(n) * 1000 + 3000, 0, None),
            c.hydro_fcst: 2_000 + rng.standard_normal(n) * 200,
            c.wind_act: 8_000 + rng.standard_normal(n) * 500,
            c.solar_act: np.clip(rng.standard_normal(n) * 1000 + 3000, 0, None),
            c.timestamp: idx,
        },
        index=idx,
    )
    df[c.residual_load_fcst] = (
        df[c.load_fcst] - df[c.wind_fcst] - df[c.solar_fcst] - df[c.hydro_fcst]
    )
    vdf = validate_hourly_frame(df, cfg)
    p95 = float(np.percentile(np.abs(vdf[c.price_da].values), 95))

    def _build(frame: pd.DataFrame):
        return build_feature_matrix(frame, cfg, train_abs_price_p95=p95)

    assert_truncated_history_matches_tail(_build, vdf, tail_start=200)


def test_lightgbm_deterministic_cpu() -> None:
    pytest.importorskip("lightgbm")
    import inspect

    from lightgbm import LGBMRegressor

    if "deterministic" not in inspect.signature(LGBMRegressor.__init__).parameters:
        pytest.skip("LightGBM build without deterministic flag")
    X = pd.DataFrame({"a": np.arange(200, dtype=float), "b": np.random.default_rng(1).random(200)})
    y = X["a"] * 0.1 + X["b"]
    p = dict(
        n_estimators=80,
        num_leaves=16,
        learning_rate=0.05,
        random_state=42,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbose=-1,
    )
    m1 = LGBMRegressor(**p).fit(X, y)
    m2 = LGBMRegressor(**p).fit(X, y)
    p1 = m1.predict(X.iloc[:20])
    p2 = m2.predict(X.iloc[:20])
    np.testing.assert_array_equal(p1, p2)
