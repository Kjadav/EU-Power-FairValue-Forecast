"""Tests for plot generation — verify PNGs and CSVs are created."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_pipeline_utils.market.adapter import ForecastSnapshot
from trading_pipeline_utils.market.curve import TranslatedProduct, TranslationDiagnostics
from trading_pipeline_utils.reporting.plots import generate_all_plots
from trading_pipeline_utils.market.signals import Signal

pytest.importorskip("matplotlib")


def _make_test_data():
    rng = np.random.default_rng(42)
    n_hours = 24
    n_paths = 50
    idx = pd.date_range("2026-04-13T00:00:00", periods=168, freq="h", tz="UTC")
    idx24 = idx[:n_hours]

    point = pd.DataFrame(
        {
            "price_mean": 80 + rng.normal(0, 3, n_hours),
            "price_p10": 70 + rng.normal(0, 3, n_hours),
            "price_p50": 80 + rng.normal(0, 3, n_hours),
            "price_p90": 90 + rng.normal(0, 3, n_hours),
        },
        index=idx24,
    )
    paths = np.column_stack([80 + rng.normal(0, 3, 168) for _ in range(n_paths)])
    paths_df = pd.DataFrame(paths, index=idx, columns=[f"path_{i}" for i in range(n_paths)])

    bundle = ForecastSnapshot(
        point_forecasts=point,
        scenario_paths=paths_df,
        backtest_summary={"overall_mae_eur_mwh": 10.0},
        oof_predictions=None,
        aggregation_level="scenario_paths",
        forecast_run_time=pd.Timestamp.now(tz="UTC"),
        as_of_date="2026-04-13",
        delivery_timezone="Europe/Berlin",
        market_name="DE-LU",
    )

    def _prod(code, fv, mkt, std=3.0):
        return TranslatedProduct(
            product_code=code, strip_type="base",
            market_forward=mkt, fair_value=fv, std=std,
            p10=fv - 10, p50=fv, p90=fv + 10,
            edge=fv - mkt, risk_premium_proxy=mkt - fv,
            diagnostics=TranslationDiagnostics(translation_quality="direct"),
        )

    products = {
        "prompt_week_base": _prod("prompt_week_base", 82.0, 80.0),
    }

    signals = {
        code: Signal(
            product_code=code, edge=p.edge, edge_z=p.edge / max(p.std, 0.01),
            risk_premium_proxy=p.risk_premium_proxy, confidence=0.6,
            signal_score=0.6 if p.edge > 0 else -0.6, direction="long" if p.edge > 0 else "short",
            suggested_position_units=1.0, suggested_expression="test",
            model_skill_weight=0.7, translation_stability_weight=0.9,
            coverage_weight=0.8, distribution_weight=0.7,
        )
        for code, p in products.items()
    }

    return bundle, signals


class TestGenerateAllPlots:
    def test_both_plots_created(self, tmp_path):
        bundle, signals = _make_test_data()
        paths = generate_all_plots(bundle, signals, tmp_path)
        assert len(paths) == 2
        for name, path in paths.items():
            assert path.exists(), f"Plot {name} not created at {path}"
            assert path.suffix == ".png"

    def test_csv_exports_created(self, tmp_path):
        bundle, signals = _make_test_data()
        generate_all_plots(bundle, signals, tmp_path)

        expected_csvs = [
            "da_hourly_forecast.csv",
            "weekly_distribution.csv",
        ]
        for csv_name in expected_csvs:
            csv_path = tmp_path / csv_name
            assert csv_path.exists(), f"CSV {csv_name} not created"
            df = pd.read_csv(csv_path)
            assert len(df) > 0, f"CSV {csv_name} is empty"
