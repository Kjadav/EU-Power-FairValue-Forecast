"""Tests for post-model curve translation and product fair value engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_pipeline_utils.market.adapter import ForecastSnapshot
from trading_pipeline_utils.settings import PostModelConfig
from trading_pipeline_utils.market.curve import (
    ProductFairValue,
    TranslationDiagnostics,
    build_curve_translation,
    compute_product_fair_values_pathwise,
    compute_product_fair_values_point,
    translate_product,
    _fit_ols_fallback,
)


def _make_bundle(
    n_hours: int = 168,
    n_paths: int = 100,
    base_price: float = 80.0,
    with_paths: bool = True,
    market_forwards: dict | None = None,
) -> ForecastSnapshot:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2026-04-13T00:00:00", periods=n_hours, freq="h", tz="UTC")

    prices = base_price + rng.normal(0, 5, size=n_hours)
    point = pd.DataFrame(
        {
            "price_mean": prices,
            "price_p10": prices - 10,
            "price_p50": prices,
            "price_p90": prices + 10,
        },
        index=idx,
    )

    paths_df = None
    agg = "point_only"
    if with_paths:
        paths = np.column_stack([
            base_price + rng.normal(0, 5, size=n_hours) for _ in range(n_paths)
        ])
        paths_df = pd.DataFrame(
            paths,
            index=idx,
            columns=[f"path_{i}" for i in range(n_paths)],
        )
        agg = "scenario_paths"

    return ForecastSnapshot(
        point_forecasts=point,
        scenario_paths=paths_df,
        backtest_summary={"overall_mae_eur_mwh": 10.0, "q10_q90_coverage": 0.78},
        oof_predictions=None,
        aggregation_level=agg,
        forecast_run_time=pd.Timestamp.now(tz="UTC"),
        as_of_date="2026-04-13",
        delivery_timezone="Europe/Berlin",
        market_name="DE-LU",
        market_forwards=market_forwards or {},
    )


class TestPathwiseFairValues:
    def test_pathwise_returns_base_and_peak(self):
        bundle = _make_bundle(n_hours=168, n_paths=50)
        config = PostModelConfig()
        fvs = compute_product_fair_values_pathwise(bundle, config)
        assert "prompt_week_base" in fvs
        assert "prompt_week_peak" in fvs

    def test_pathwise_fv_is_mean_of_path_means(self):
        bundle = _make_bundle(n_hours=168, n_paths=200)
        config = PostModelConfig()
        fvs = compute_product_fair_values_pathwise(bundle, config)
        base = fvs["prompt_week_base"]

        paths_matrix = bundle.scenario_paths[[c for c in bundle.scenario_paths.columns if c.startswith("path_")]].values
        expected_mean = float(np.mean(paths_matrix.mean(axis=0)))
        assert abs(base.fair_value_mean - expected_mean) < 1e-6

    def test_never_averages_hourly_quantiles_for_product(self):
        """Product quantiles must come from pathwise FV distribution, not hourly quantile averaging."""
        bundle = _make_bundle(n_hours=168, n_paths=200)
        config = PostModelConfig()
        fvs = compute_product_fair_values_pathwise(bundle, config)
        base = fvs["prompt_week_base"]

        hourly_p10_avg = float(bundle.point_forecasts["price_p10"].mean())
        assert base.fair_value_p10 is not None
        assert base.method == "direct_pathwise"
        assert base.n_paths == 200

    def test_pathwise_raises_without_paths(self):
        bundle = _make_bundle(with_paths=False)
        config = PostModelConfig()
        with pytest.raises(ValueError, match="Scenario paths required"):
            compute_product_fair_values_pathwise(bundle, config)


class TestPointFairValues:
    def test_point_returns_next_day(self):
        bundle = _make_bundle(n_hours=24, with_paths=False)
        config = PostModelConfig()
        fvs = compute_product_fair_values_point(bundle, config)
        assert "next_day_base" in fvs
        assert fvs["next_day_base"].method == "point_mean"


class TestTranslationRegression:
    def test_ols_fallback_basic(self):
        rng = np.random.default_rng(99)
        n = 200
        dw = rng.normal(0, 1, n)
        dp = 0.5 + 0.8 * dw + rng.normal(0, 0.1, n)
        alpha, beta, resid_std, r2 = _fit_ols_fallback(dw, dp)
        assert abs(beta - 0.8) < 0.1
        assert abs(alpha - 0.5) < 0.1
        assert r2 > 0.8

    def test_ols_insufficient_data_returns_defaults(self):
        alpha, beta, resid_std, r2 = _fit_ols_fallback(
            np.array([1.0, 2.0]), np.array([3.0, 4.0])
        )
        assert beta == 1.0
        assert r2 == 0.0


class TestTranslateProduct:
    def test_edge_sign_positive_when_fv_above_market(self):
        week_fv = ProductFairValue("prompt_week_base", "base", fair_value_mean=90.0, fair_value_std=3.0)
        result = translate_product(
            "prompt_month_base", "base",
            market_forward_product=82.0,
            market_forward_week=85.0,
            model_fv_week=week_fv,
            config=PostModelConfig(),
        )
        assert result.edge > 0
        assert result.risk_premium_proxy < 0

    def test_edge_sign_negative_when_fv_below_market(self):
        week_fv = ProductFairValue("prompt_week_base", "base", fair_value_mean=80.0, fair_value_std=3.0)
        result = translate_product(
            "prompt_month_base", "base",
            market_forward_product=88.0,
            market_forward_week=85.0,
            model_fv_week=week_fv,
            config=PostModelConfig(),
        )
        assert result.edge < 0
        assert result.risk_premium_proxy > 0

    def test_insufficient_history_raises_without_fallback(self):
        week_fv = ProductFairValue("prompt_week_base", "base", fair_value_mean=90.0)
        config = PostModelConfig()
        config.translation.enable_fallback_shrinkage = False
        config.translation.min_history_days = 60

        with pytest.raises(ValueError, match="Insufficient history"):
            translate_product(
                "prompt_month_base", "base",
                market_forward_product=85.0,
                market_forward_week=85.0,
                model_fv_week=week_fv,
                config=config,
                historical_delta_week=np.zeros(10),
                historical_delta_product=np.zeros(10),
            )


class TestBuildCurveTranslation:
    def test_direct_products_included(self):
        bundle = _make_bundle(n_hours=168, n_paths=50)
        config = PostModelConfig()
        products = build_curve_translation(bundle, config)
        assert "prompt_week_base" in products
        assert "prompt_week_peak" in products

    def test_translated_products_with_market_forwards(self):
        fwd = {
            "prompt_week_base": 80.0,
            "prompt_week_peak": 95.0,
            "prompt_month_base": 82.0,
            "prompt_month_peak": 97.0,
        }
        bundle = _make_bundle(n_hours=168, n_paths=50, market_forwards=fwd)
        config = PostModelConfig()
        products = build_curve_translation(bundle, config)
        assert "prompt_month_base" in products
        assert "prompt_month_peak" in products

    def test_dst_safe_peak_classification(self):
        """Peak mask uses local timezone, not UTC."""
        idx = pd.date_range("2026-03-29T00:00:00", periods=168, freq="h", tz="UTC")
        rng = np.random.default_rng(42)
        paths = np.column_stack([80 + rng.normal(0, 3, 168) for _ in range(20)])
        paths_df = pd.DataFrame(paths, index=idx, columns=[f"path_{i}" for i in range(20)])
        point = pd.DataFrame(
            {"price_mean": 80.0, "price_p10": 70.0, "price_p50": 80.0, "price_p90": 90.0},
            index=idx,
        )
        bundle = ForecastSnapshot(
            point_forecasts=point,
            scenario_paths=paths_df,
            backtest_summary={},
            oof_predictions=None,
            aggregation_level="scenario_paths",
            forecast_run_time=pd.Timestamp.now(tz="UTC"),
            as_of_date="2026-03-29",
            delivery_timezone="Europe/Berlin",
            market_name="DE-LU",
        )
        config = PostModelConfig()
        fvs = compute_product_fair_values_pathwise(bundle, config)
        peak = fvs.get("prompt_week_peak")
        assert peak is not None
        assert peak.n_hours > 0
        assert peak.n_hours < 168
