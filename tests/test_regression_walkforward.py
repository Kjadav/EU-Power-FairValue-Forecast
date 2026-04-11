"""Synthetic bundle: walk-forward OLS + naive backtest must complete."""

from __future__ import annotations

from pathlib import Path

import pytest

from backtest.regression.synthetic_bundle import write_synthetic_regression_bundle
from backtest.regression.walkforward import walk_forward_day_ahead_backtest
from eu_power_forecast.forecasting.regression import fit_ols_beta, load_day_ahead_regression_panel


def test_walk_forward_backtest_completes(tmp_path: Path) -> None:
    b = tmp_path / "bundle"
    write_synthetic_regression_bundle(b)
    out = walk_forward_day_ahead_backtest(b, n_eval_days=3, min_train_hours=150)
    assert out["n_eval_days_completed"] >= 1
    assert "forecast_frame" in out
    assert list(out["forecast_frame"].columns) == [
        "actual",
        "regression",
        "naive_last",
        "naive_seasonal_24h",
    ]
    om = out["overall_metrics"]
    for k in ("regression", "naive_last", "naive_seasonal_24h"):
        assert om[k]["n"] > 0
        assert om[k]["mae"] >= 0
        assert om[k]["rmse"] >= 0


def test_forecasting_regression_module_path() -> None:
    assert callable(fit_ols_beta)
    assert callable(load_day_ahead_regression_panel)


def test_deploy_regression_backtest_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from backtest.regression.pipeline import deploy_regression_backtest

    b = tmp_path / "bundle"
    write_synthetic_regression_bundle(b)
    art = deploy_regression_backtest(
        b,
        n_eval_days=2,
        min_train_hours=150,
        print_summary=False,
        json_path=None,
    )
    assert "result" in art and "overall_df" in art and "per_day_df" in art and "figure" in art
    assert "forecast_frame" in art and len(art["forecast_frame"]) > 0
    assert art["fallback_note"] is None
    assert len(art["overall_df"]) == 3


def test_short_bundle_raises_clear_error(tmp_path: Path) -> None:
    b = tmp_path / "short"
    write_synthetic_regression_bundle(b, n_hours=48)
    with pytest.raises(ValueError, match="P_{t-168}"):
        walk_forward_day_ahead_backtest(b, n_eval_days=1, min_train_hours=10)
