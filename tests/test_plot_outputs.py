"""Legacy plot smoke test — behaviour covered in tests/unit/test_plots.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_pipeline_utils.market.adapter import ForecastSnapshot
from trading_pipeline_utils.market.signals import Signal
from trading_pipeline_utils.reporting.plots import generate_all_plots

pytest.importorskip("matplotlib")


def _make_bundle_and_signals():
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
    sig = Signal(
        product_code="prompt_week_base",
        edge=2.0,
        edge_z=0.67,
        risk_premium_proxy=-2.0,
        confidence=0.5,
        signal_score=0.5,
        direction="long",
        suggested_position_units=1.0,
        suggested_expression="test",
        model_skill_weight=0.7,
        translation_stability_weight=0.9,
        coverage_weight=0.8,
        distribution_weight=0.7,
    )
    return bundle, {"prompt_week_base": sig}


def test_generate_all_plots_current_api(tmp_path):
    bundle, signals = _make_bundle_and_signals()
    paths = generate_all_plots(bundle, signals, tmp_path)
    assert len(paths) == 2
    assert (tmp_path / "da_hourly_forecast.png").exists()
    assert (tmp_path / "weekly_distribution.png").exists()
