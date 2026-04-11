from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from data.fetcher import DataPayload, fetch
from data.validator import ValidationResult, validate as stat_validate
from llm.validator import LLMValidationResult, validate as llm_validate_fn
from models.base import forecast as naive_forecast
from models.forecaster import (
    PRICE_COL,
    add_price_lags,
    backtest_report,
    build_design_xy,
    build_panel_from_payload,
    fit_ols,
    predict_ols,
    row_mask_valid,
)
from signals.generator import Signal, generate as generate_signal
from utils.logging import step_timer

logger = logging.getLogger(__name__)


def fetch_data(config: dict[str, Any]) -> DataPayload:
    with step_timer("fetch_data", logger) as ctx:
        data = fetch(config)
        rows = sum(len(df) for _, df in data.iter_core_tables())
        logger.info(
            "fetched %d tables, %d total rows, region=%s",
            len(data.iter_core_tables()), rows, data.region,
        )
        ctx["rows"] = rows
    return data


def validate_data(data: DataPayload, config: dict[str, Any]) -> ValidationResult:
    with step_timer("validate_data", logger) as ctx:
        result = stat_validate(data, config)
        pass_count = sum(
            1 for v in result.tables.values()
            if v.get("hourly_utc", {}).get("ok") and v.get("non_negative", {}).get("ok")
        )
        logger.info(
            "validation: %d/%d tables passed, overall_ok=%s",
            pass_count, len(result.tables), result.ok,
        )
        if not result.ok:
            logger.warning("statistical validation flagged issues")
        ctx["ok"] = result.ok
    return result


def llm_validate_data(data: DataPayload, config: dict[str, Any]) -> LLMValidationResult:
    with step_timer("llm_validate_data", logger) as ctx:
        if not config.get("pipeline", {}).get("llm_qa", False):
            logger.info("LLM QA disabled, skipping")
            result = LLMValidationResult(
                verdict="pass", confidence=1.0, issues=[], details={"skipped": True},
            )
            ctx["verdict"] = "skipped"
            return result
        stat_result = stat_validate(data, config)
        result = llm_validate_fn(data, stat_result, config)
        logger.info(
            "LLM verdict=%s, confidence=%.2f, issues=%d",
            result.verdict, result.confidence, len(result.issues),
        )
        for issue in result.issues:
            logger.warning("LLM issue: %s", issue)
        ctx["verdict"] = result.verdict
    return result


def run_base_model(data: DataPayload, config: dict[str, Any]) -> dict[str, Any]:
    with step_timer("run_base_model", logger) as ctx:
        model_cfg = config.get("model", {})
        method = model_cfg.get("baseline_method", "last")
        horizon = model_cfg.get("forecast_horizon", 24)
        baseline = naive_forecast(data.day_ahead_prices, horizon, method=method)
        logger.info(
            "baseline: method=%s, horizon=%d, mean=%.2f",
            method, horizon, float(baseline.mean()),
        )
        ctx["method"] = method
    return {"forecast": baseline, "method": method, "horizon": horizon}


def run_forecasting(
    model: dict[str, Any],
    data: DataPayload,
    config: dict[str, Any],
) -> dict[str, Any]:
    with step_timer("run_forecasting", logger) as ctx:
        panel = build_panel_from_payload(data)
        panel = add_price_lags(panel)
        X, y = build_design_xy(panel)
        mask = row_mask_valid(X, y)
        X_clean, y_clean = X[mask], y[mask]

        horizon = model["horizon"]
        baseline_vals = model["forecast"].values

        if len(X_clean) < 48:
            logger.warning(
                "insufficient data for OLS (%d rows), returning baseline only",
                len(X_clean),
            )
            return {
                "forecast": baseline_vals,
                "baseline": baseline_vals,
                "metrics": {"n": 0, "mae": float("nan"), "rmse": float("nan")},
                "signal": Signal("NEUTRAL", "Uncertain", 0.0, "insufficient data"),
            }

        beta = fit_ols(X_clean, y_clean)
        y_hat = predict_ols(X_clean, beta)
        y_hat_series = pd.Series(y_hat, index=y_clean.index)
        metrics = backtest_report(y_clean, y_hat_series)

        n_fwd = min(horizon, len(y_hat))
        model_fwd = y_hat[-n_fwd:]
        baseline_cmp = baseline_vals[:n_fwd]

        signal = generate_signal(baseline_cmp, model_fwd, metrics, config)
        logger.info(
            "OLS: n=%d, MAE=%.2f, RMSE=%.2f",
            metrics["n"], metrics["mae"], metrics["rmse"],
        )
        logger.info(
            "signal: %s / %s (conf=%.2f)",
            signal.direction, signal.outlook, signal.confidence,
        )
        ctx["mae"] = metrics["mae"]
        ctx["signal"] = signal.direction

    return {
        "forecast": model_fwd,
        "baseline": baseline_cmp,
        "metrics": metrics,
        "signal": signal,
    }
