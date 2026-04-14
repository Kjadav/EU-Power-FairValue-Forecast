"""Smoke test: verifies the post-model analytics pipeline runs end-to-end."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_pipeline_utils.market.adapter import ForecastSnapshot, load_forecast_snapshot
from trading_pipeline_utils.market.curve import build_curve_translation
from trading_pipeline_utils.market.signals import compute_all_signals
from trading_pipeline_utils.market.invalidation import evaluate_all_invalidations
from trading_pipeline_utils.llm.summarizer import build_signal_payload, generate_llm_insights
from trading_pipeline_utils.reporting.tables import build_deterministic_summaries
from trading_pipeline_utils.settings import PostModelConfig


def _create_synthetic_artifacts(tmpdir: Path) -> Path:
    """Create minimal synthetic forecast artifacts for testing."""
    idx_24 = pd.date_range("2026-04-12", periods=24, freq="h", tz="UTC")
    fc = pd.DataFrame({
        "y_point": np.random.uniform(40, 80, 24),
        "q10": np.random.uniform(30, 50, 24),
        "q50": np.random.uniform(45, 75, 24),
        "q90": np.random.uniform(70, 100, 24),
    }, index=idx_24)
    fc.to_parquet(tmpdir / "final_forecasts.parquet")

    idx_168 = pd.date_range("2026-04-12", periods=168, freq="h", tz="UTC")
    paths = pd.DataFrame(
        np.random.uniform(40, 80, (168, 100)),
        index=idx_168,
        columns=[f"path_{i}" for i in range(100)],
    )
    paths.to_parquet(tmpdir / "simulated_paths.parquet")

    bt = pd.DataFrame([{
        "overall_mae_eur_mwh": 10.5,
        "overall_rmse_eur_mwh": 22.0,
        "q10_q90_coverage": 0.65,
        "n_oof_hours": 5000,
        "n_folds": 30,
    }])
    bt.to_csv(tmpdir / "backtest_summary.csv", index=False)
    return tmpdir


def test_full_postmodel_pipeline():
    """Full post-model analytics pipeline with synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir) / "artifacts"
        artifact_dir.mkdir()
        _create_synthetic_artifacts(artifact_dir)

        config = PostModelConfig()
        config.llm_insight.enabled = False

        market_forwards = {
            "prompt_week_base": 65.0,
            "prompt_week_peak": 80.0,
            "prompt_month_base": 67.0,
            "prompt_month_peak": 82.0,
        }

        bundle = load_forecast_snapshot(artifact_dir, config, market_forwards=market_forwards)
        assert len(bundle.point_forecasts) == 24
        assert bundle.scenario_paths is not None

        products = build_curve_translation(bundle, config)
        assert len(products) >= 2

        signals = compute_all_signals(products, bundle.backtest_summary, config)
        assert len(signals) == len(products)

        invalidations = evaluate_all_invalidations(signals, products, config)
        assert len(invalidations) == len(signals)

        summaries = build_deterministic_summaries(signals, products, invalidations)
        assert len(summaries) >= 2

        payload = build_signal_payload(
            products, signals, [], invalidations,
            bundle.backtest_summary, config, bundle.as_of_date,
        )
        assert "outright_signals" in payload
        assert json.dumps(payload, default=str)

        insight = generate_llm_insights(payload, config)
        assert insight.source == "deterministic_fallback"
        assert len(insight.executive_summary) > 0


def test_config_loading():
    """Verify config loads from default pipeline.yaml."""
    from trading_pipeline_utils.settings import load_config, default_config_path
    path = default_config_path()
    if path.exists():
        config = load_config(path)
        assert config.data_vendor.type == "smard"


def test_dispatcher_imports():
    """Verify all dispatcher functions are importable."""
    from dispatcher import (
        fetch_data,
        generate_llm_summary,
        generate_reports,
        generate_signals,
        persist_outputs,
        run_invalidations,
        run_llm_data_qa,
        translate_curve,
        validate_inputs,
    )


def test_exports_run_directory():
    """Verify run directory creation and symlink."""
    from trading_pipeline_utils.reporting.exports import create_run_directory, update_latest_symlink
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        run_dir = create_run_directory(base)
        assert run_dir.is_dir()
        assert "runs" in str(run_dir)
