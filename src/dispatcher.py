from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from trading_pipeline_utils.data.fetch import fetch_smard_bundle
from trading_pipeline_utils.data.schemas import DataPayload, SmardData, ValidationResult
from trading_pipeline_utils.llm.data_qa import validate as llm_validate_bundle
from trading_pipeline_utils.llm.schemas import LLMInsightResult, LLMValidationResult
from trading_pipeline_utils.llm.summarizer import build_signal_payload, generate_llm_insights
from trading_pipeline_utils.logging_config import step_timer
from trading_pipeline_utils.market.adapter import ForecastSnapshot, load_forecast_snapshot
from trading_pipeline_utils.market.curve import TranslatedProduct, build_curve_translation
from trading_pipeline_utils.market.invalidation import InvalidationResult, evaluate_all_invalidations
from trading_pipeline_utils.market.signals import Signal, compute_all_signals
from trading_pipeline_utils.reporting.exports import (
    create_run_directory,
    save_config_snapshot,
    save_dataframe,
    save_json,
    save_text,
    update_latest_symlink,
    write_run_summary,
)
from trading_pipeline_utils.reporting.plots import generate_all_plots
from trading_pipeline_utils.reporting.tables import build_deterministic_summaries
from trading_pipeline_utils.settings import PipelineConfig, PostModelConfig
from trading_pipeline_utils.validation.checks import attach_validation, validate_data_payload

logger = logging.getLogger(__name__)


def fetch_data(config: PipelineConfig) -> DataPayload:
    """Fetch market data from SMARD API or load from existing bundle."""
    with step_timer(logger, "fetch_data"):
        payload = fetch_smard_bundle(config)
    n = sum(len(d) for _n, d in payload.bundle.iter_core_tables())
    logger.info("fetch_data: %d total rows, vendor=%s", n, payload.vendor)
    return payload


def load_existing_bundle(bundle_dir: Path, config: PipelineConfig) -> DataPayload:
    """Load an existing SMARD bundle from disk."""
    with step_timer(logger, "load_existing_bundle"):
        bundle = SmardData.load(bundle_dir, config.data_vendor.meta_filename)
    return DataPayload(bundle=bundle, vendor="smard")


def validate_inputs(data: DataPayload, config: PipelineConfig) -> ValidationResult:
    """Run deterministic data validation checks."""
    with step_timer(logger, "validate_inputs"):
        result = validate_data_payload(data, config)
        attach_validation(data, result)
    logger.info("validate_inputs: ok=%s", result.ok)
    if not result.ok:
        logger.warning("Deterministic validation reported failures")
    return result


def run_llm_data_qa(
    data: DataPayload,
    validation: ValidationResult,
    config: PipelineConfig,
) -> LLMValidationResult:
    """Optional LLM-based data quality assessment."""
    with step_timer(logger, "llm_data_qa"):
        result = llm_validate_bundle(data, validation, config)
    logger.info("llm_data_qa: verdict=%s, confidence=%.2f", result.verdict, result.confidence)
    return result


def run_forecasts(
    bundle_dir: Path,
    models_path: Path,
    out_dir: Path,
    *,
    allow_irregular_hourly: bool = True,
) -> dict[str, Any]:
    """Run next-day forecast and next-week simulation using trained models."""
    from trading_pipeline_utils.models.inference import run_forecast_next_day, run_simulate_next_week

    with step_timer(logger, "run_forecasts"):
        fc = run_forecast_next_day(
            bundle_dir, models_path, out_dir=out_dir,
            allow_irregular_hourly=allow_irregular_hourly,
        )
        sim = run_simulate_next_week(
            bundle_dir, models_path, out_dir=out_dir,
            allow_irregular_hourly=allow_irregular_hourly,
        )
    n_paths = sim.get("paths", [])
    n_paths_n = int(n_paths.shape[0]) if hasattr(n_paths, "shape") else 0
    logger.info("run_forecasts: %d forecast rows, %d simulation paths", len(fc), n_paths_n)
    return {"forecasts": fc, "simulation": sim}


def translate_curve(
    bundle: ForecastSnapshot,
    config: PostModelConfig,
) -> dict[str, TranslatedProduct]:
    """Translate hourly forecasts into tradable product fair values."""
    with step_timer(logger, "translate_curve"):
        products = build_curve_translation(bundle, config)
    logger.info("translate_curve: %d products", len(products))
    return products


def generate_signals(
    products: dict[str, TranslatedProduct],
    backtest_summary: dict[str, Any],
    config: PostModelConfig,
) -> dict[str, Signal]:
    """Compute confidence-weighted outright signals."""
    with step_timer(logger, "generate_signals"):
        signals = compute_all_signals(products, backtest_summary, config)
    logger.info("generate_signals: %s", {c: s.direction for c, s in signals.items()})
    return signals


def run_invalidations(
    signals: dict[str, Signal],
    products: dict[str, TranslatedProduct],
    config: PostModelConfig,
    *,
    prior_fair_values: dict[str, float] | None = None,
) -> dict[str, InvalidationResult]:
    """Evaluate deterministic invalidation rules."""
    with step_timer(logger, "run_invalidations"):
        inv = evaluate_all_invalidations(signals, products, config, prior_fair_values=prior_fair_values)
    n_flagged = sum(1 for v in inv.values() if v.invalidation_flag)
    logger.info("run_invalidations: %d/%d flagged", n_flagged, len(inv))
    return inv


def generate_reports(
    run_dir: Path,
    bundle: ForecastSnapshot,
    signals: dict[str, Signal],
    products: dict[str, TranslatedProduct],
    invalidations: dict[str, InvalidationResult],
    *,
    actual_prices: pd.Series | None = None,
) -> dict[str, Any]:
    """Generate output artifacts: 2 plots + submission.csv + signal summary."""
    with step_timer(logger, "generate_reports"):
        summaries = build_deterministic_summaries(signals, products, invalidations)
        save_text(run_dir, "signal_summary.txt", summaries)

        plot_paths = generate_all_plots(
            bundle, signals, run_dir,
            actual_prices=actual_prices,
        )

        submission = build_submission_csv(bundle)
        save_dataframe(run_dir, "submission.csv", submission, fmt="csv")

    logger.info("generate_reports: %d plots, submission.csv (%d rows)", len(plot_paths), len(submission))
    return {
        "summaries": summaries,
        "plot_paths": plot_paths,
        "submission_rows": len(submission),
    }


def build_submission_csv(bundle: ForecastSnapshot) -> pd.DataFrame:
    """Build submission.csv with out-of-sample predictions: id, y_pred."""
    fc = bundle.point_forecasts
    rows = []
    for ts in fc.index:
        ts_str = ts.strftime("%Y-%m-%d_%H:%M")
        rows.append({
            "id": f"{ts_str}_da_price",
            "y_pred": round(float(fc.loc[ts, "price_mean"]), 4),
        })

    if bundle.scenario_paths is not None:
        paths_df = bundle.scenario_paths
        path_cols = [c for c in paths_df.columns if c.startswith("path_")]
        hourly_mean = paths_df[path_cols].mean(axis=1)
        for ts in paths_df.index:
            if ts not in fc.index:
                ts_str = ts.strftime("%Y-%m-%d_%H:%M")
                rows.append({
                    "id": f"{ts_str}_da_price",
                    "y_pred": round(float(hourly_mean.loc[ts]), 4),
                })

    df = pd.DataFrame(rows)
    df = df.sort_values("id").reset_index(drop=True)
    return df


def generate_llm_summary(
    run_dir: Path,
    products: dict[str, TranslatedProduct],
    signals: dict[str, Signal],
    invalidations: dict[str, InvalidationResult],
    backtest_summary: dict[str, Any],
    config: PostModelConfig,
    as_of_date: str,
) -> LLMInsightResult:
    """Build LLM payload and generate structured desk commentary."""
    with step_timer(logger, "generate_llm_summary"):
        payload = build_signal_payload(
            products, signals, [], invalidations,
            backtest_summary, config, as_of_date,
        )
        save_json(run_dir, "llm_payload.json", payload)

        insight = generate_llm_insights(payload, config)
        save_json(run_dir, "llm_insight.json", asdict(insight))

    logger.info("generate_llm_summary: source=%s", insight.source)
    return insight


def persist_outputs(
    run_dir: Path,
    results_dir: Path,
    stages: dict[str, str],
    *,
    config: Any = None,
    llm_insight: LLMInsightResult | None = None,
) -> None:
    """Write final run summary and update ``results/latest`` symlink."""
    with step_timer(logger, "persist_outputs"):
        if config is not None:
            save_config_snapshot(run_dir, config)
        write_run_summary(run_dir, stages, llm_insight=llm_insight)
        update_latest_symlink(results_dir, run_dir)
    logger.info("persist_outputs: %s", run_dir)
