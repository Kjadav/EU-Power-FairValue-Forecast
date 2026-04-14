from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dispatcher import (
    fetch_data,
    generate_llm_summary,
    generate_reports,
    generate_signals,
    persist_outputs,
    run_forecasts,
    run_invalidations,
    translate_curve,
    validate_inputs,
)
from trading_pipeline_utils.logging_config import configure_logging
from trading_pipeline_utils.market.adapter import load_forecast_snapshot
from trading_pipeline_utils.reporting.exports import create_run_directory
from trading_pipeline_utils.settings import PostModelConfig, load_config
from trading_pipeline_utils.utils.paths import default_results_dir

logger = logging.getLogger(__name__)


def _compute_reference_forwards(
    bundle_dir: Path,
    delivery_tz: str = "Europe/Berlin",
    peak_start: int = 8,
    peak_end: int = 20,
) -> dict[str, float]:
    """Derive reference forwards from the last realized week of DA prices.

    When no market forwards are supplied, use the most recent 168 hours of
    realized prices as the reference.  This gives the signal framework a
    meaningful comparison: "is the model forecasting higher or lower than
    the most recent realized week?"
    """
    price_path = Path(bundle_dir) / "day_ahead_prices.parquet"
    if not price_path.is_file():
        return {}
    df = pd.read_parquet(price_path)
    if df.empty:
        return {}

    prices = df.iloc[:, 0].dropna()
    if len(prices) < 168:
        return {}

    last_week = prices.iloc[-168:]
    base_avg = float(last_week.mean())

    idx = last_week.index.tz_convert(delivery_tz) if last_week.index.tz is not None else last_week.index
    peak_mask = np.array(
        [(d.dayofweek < 5) and (peak_start <= d.hour < peak_end) for d in idx],
        dtype=bool,
    )
    peak_avg = float(last_week.values[peak_mask].mean()) if peak_mask.any() else base_avg

    logger.info(
        "Reference forwards from last realized week: base=%.2f, peak=%.2f EUR/MWh",
        base_avg, peak_avg,
    )
    return {
        "prompt_week_base": base_avg,
        "prompt_week_peak": peak_avg,
    }


def _load_dotenv() -> None:
    """Load .env file into environment."""
    for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]:
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.replace("export ", "").strip()
            if not key or key in os.environ:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ[key] = value
        break


def run_pipeline() -> int:
    """
    MAIN function to run the entire piepline end-to-end
    The different steps that follows in this function are:
        1. Load config
        2. Initialize logging
        3. Fetch data
        4. Validate data
        5. Run model inference
        6. curvve translation + signals 
        7. Run invalidation checks
        8. Generate plots, submission.csv, signal summary
        9. Generate LLM summary
        10. store results in results
    """
    _load_dotenv()

    parser = argparse.ArgumentParser(
        prog="trading-pipeline",
        description="European power day-ahead trading pipeline",
    )
    parser.parse_args()

    config = load_config(None)
    logger.info("Pipeline started")

    results_dir = default_results_dir()
    run_dir = create_run_directory(results_dir)
    logger.info("Run directory: %s", run_dir)

    stages: dict[str, str] = {}
    model_dir = Path("data/processed/da_model")
    bundle_dir = Path(config.pipeline.output_dir)

    # --- Fetching data from smard.de--
    try:
        data = fetch_data(config)
        bundle_dir = Path(config.pipeline.output_dir)
        stages["fetch_data"] = f"ok ({sum(len(d) for _, d in data.bundle.iter_core_tables())} rows)"
    except Exception as e:
        logger.error("Data fetch failed: %s", e)
        stages["fetch_data"] = f"failed: {e}"
        if not bundle_dir.exists():
            logger.error("No existing fetched data to use instead")
            return None

    # --- Validate data ---
    try:
        from trading_pipeline_utils.data.schemas import SmardData, DataPayload
        bundle = SmardData.load(bundle_dir, config.data_vendor.meta_filename)
        data = DataPayload(bundle=bundle, vendor="smard")
        validation = validate_inputs(data, config)
        stages["validate_data"] = f"ok={validation.ok}"
    except Exception as e:
        logger.warning("Validation failed: %s — continuing process", e)
        stages["validate_data"] = f"warning: {e}"
        validation = None

    # --- Run forecasts ---
    models_path = model_dir / "fitted_models.joblib"
    if models_path.is_file():
        try:
            run_forecasts(
                bundle_dir,
                models_path,
                model_dir,
                allow_irregular_hourly=True,
            )
            stages["run_forecasts"] = "ok"
        except Exception as e:
            logger.warning("Forecast failed: %s — using existing model outputs", e)
            stages["run_forecasts"] = f"warning: {e}"
    else:
        logger.info("No fitted models at %s — using existing forecast files", models_path)
        stages["run_forecasts"] = "skipped (no models)"

    # --- Curve translations + signals generation ---
    postmodel_config = PostModelConfig()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        postmodel_config.llm_insight.api_key = gemini_key

    market_forwards = _compute_reference_forwards(bundle_dir)
    if market_forwards:
        stages["reference_forwards"] = ", ".join(
            f"{k}={v:.2f}" for k, v in market_forwards.items()
        )

    try:
        fc_bundle = load_forecast_snapshot(
            model_dir, postmodel_config,
            market_forwards=market_forwards,
        )

        products = translate_curve(fc_bundle, postmodel_config)
        stages["translate_curve"] = f"{len(products)} products"

        signals = generate_signals(products, fc_bundle.backtest_summary, postmodel_config)
        stages["generate_signals"] = f"{len(signals)} signals"

        invalidations = run_invalidations(signals, products, postmodel_config)
        n_flagged = sum(1 for v in invalidations.values() if v.invalidation_flag)
        stages["invalidations"] = f"{n_flagged}/{len(invalidations)} flagged"

    except Exception as e:
        logger.error("Post-model analytics failed: %s", e)
        stages["post_model"] = f"failed: {e}"
        persist_outputs(run_dir, results_dir, stages, config=config)
        return 1

    # --- Generate reports ---
    try:
        report_result = generate_reports(
            run_dir, fc_bundle, signals, products, invalidations,
        )
        stages["generate_reports"] = "ok"
    except Exception as e:
        logger.warning("Report generation failed: %s", e)
        stages["generate_reports"] = f"warning: {e}"

    # --- Generate LLM summary ---
    try:
        llm_insight = generate_llm_summary(
            run_dir, products, signals, invalidations,
            fc_bundle.backtest_summary, postmodel_config, fc_bundle.as_of_date,
        )
        stages["llm_summary"] = f"source={llm_insight.source}"
    except Exception as e:
        logger.warning("LLM summary failed: %s", e)
        stages["llm_summary"] = f"warning: {e}"
        llm_insight = None

    # --- Persist ---
    persist_outputs(run_dir, results_dir, stages, config=config, llm_insight=llm_insight)
    stages["persist"] = "ok"

  
    print("Pipeline Run Complete")

    print(f"Output: {run_dir}")
    for stage, status in stages.items():
        print(f"  {stage:30s} {status}")
    if llm_insight:
        print(f"\nExecutive Summary:\n  {llm_insight.executive_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_pipeline())
