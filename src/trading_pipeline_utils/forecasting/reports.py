from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_pipeline_utils.forecasting.backtest import run_rolling_backtest
from trading_pipeline_utils.forecasting.loader import load_smard_bundle_hourly
from trading_pipeline_utils.models.inference import (
    load_production_artifact,
    run_forecast_next_day,
    run_simulate_next_week,
)
from trading_pipeline_utils.settings import ModelPipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forecast report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastReportArtifacts:
    next_day_hourly_csv: Path
    next_week_avg_summary_csv: Path
    next_week_avg_per_path_csv: Path
    combined_summary_csv: Path
    figure_next_day_da_png: Path
    figure_next_week_avg_distribution_png: Path


def run_forecast_report(
    bundle_dir: Path,
    models_path: Path,
    out_dir: Path,
    *,
    allow_irregular_hourly: bool = True,
) -> ForecastReportArtifacts:
    """
    Produces
    - day ahead hourly pricing
    - next-week average price from path simulation
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_production_artifact(Path(models_path))
    tz = artifact.config.validation.delivery_timezone

    fc24 = run_forecast_next_day(
        Path(bundle_dir),
        Path(models_path),
        out_dir=out_dir,
        allow_irregular_hourly=allow_irregular_hourly,
    )
    hourly_csv = out_dir / "report_next_day_hourly_da.csv"
    loc24 = fc24.index.tz_convert(tz)
    hourly_out = pd.DataFrame(
        {
            "timestamp_utc": fc24.index,
            "timestamp_local": loc24,
            "local_date": loc24.strftime("%Y-%m-%d"),
            "local_hour": loc24.hour,
            "day_ahead_price_point_eur_mwh": fc24["y_point"].values,
            "day_ahead_price_q10_eur_mwh": fc24["q10"].values,
            "day_ahead_price_q50_eur_mwh": fc24["q50"].values,
            "day_ahead_price_q90_eur_mwh": fc24["q90"].values,
        }
    )
    hourly_out.to_csv(hourly_csv, index=False)

    sim = run_simulate_next_week(
        Path(bundle_dir),
        Path(models_path),
        out_dir=out_dir,
        allow_irregular_hourly=allow_irregular_hourly,
    )
    paths: np.ndarray = sim["paths"]
    weekly_avg_per_path = np.asarray(paths.mean(axis=1), dtype=np.float64)
    if not np.isclose(float(sim["weekly_baseload_mean"]), float(np.mean(weekly_avg_per_path)), rtol=0, atol=1e-4):
        raise RuntimeError(
            "weekly_baseload_mean must match mean of per-path weekly averages"
        )

    per_path_csv = out_dir / "report_next_week_avg_da_per_simulated_path.csv"
    pd.DataFrame({"weekly_avg_da_eur_mwh": weekly_avg_per_path}).to_csv(per_path_csv, index=False)
    w_mean = float(np.mean(weekly_avg_per_path))
    w_std = float(np.std(weekly_avg_per_path))
    w_p05, w_p25, w_p50, w_p75, w_p95 = np.percentile(
        weekly_avg_per_path, [5, 25, 50, 75, 95]
    ).tolist()

    week_summary = pd.DataFrame(
        [
            {
                "metric": "next_week_avg_da_eur_mwh_from_path_distribution",
                "expectation_mean_of_path_means": w_mean,
                "std_across_paths": w_std,
                "p05": w_p05,
                "p25": w_p25,
                "p50": w_p50,
                "p75": w_p75,
                "p95": w_p95,
                "n_paths": paths.shape[0],
                "horizon_hours": paths.shape[1],
                "weekly_baseload_mean_simulation_module": sim.get("weekly_baseload_mean"),
                "weekly_baseload_p50_simulation_module": sim.get("weekly_baseload_p50"),
            }
        ]
    )
    week_csv = out_dir / "report_next_week_avg_da_summary.csv"
    week_summary.to_csv(week_csv, index=False)

    next_day_mean = float(fc24["y_point"].mean())
    next_day_q50_mean = float(fc24["q50"].mean())
    combined = pd.DataFrame(
        [
            {
                "next_24h_point_mean_eur_mwh": next_day_mean,
                "next_24h_q50_mean_eur_mwh": next_day_q50_mean,
                "next_week_expected_avg_da_eur_mwh": w_mean,
                "next_week_avg_da_p05_eur_mwh": w_p05,
                "next_week_avg_da_p50_eur_mwh": w_p50,
                "next_week_avg_da_p95_eur_mwh": w_p95,
                "delivery_timezone": tz,
            }
        ]
    )
    combined_csv = out_dir / "report_forecast_summary.csv"
    combined.to_csv(combined_csv, index=False)

    # next-day hourly day ahead pricing
    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(fc24))
    ax1.fill_between(
        x,
        fc24["q10"].values,
        fc24["q90"].values,
        alpha=0.25,
        color="C0",
        label="q10–q90",
    )
    ax1.plot(x, fc24["q50"].values, ":", color="C1", label="Median (q50)")
    ax1.plot(x, fc24["y_point"].values, "-", color="C0", linewidth=2, label="Point forecast")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{loc24[i].strftime('%m-%d %H')}h" for i in x], rotation=45, ha="right")
    ax1.set_ylabel("EUR/MWh")
    ax1.set_title(f"Next-day hourly day-ahead price ({tz})")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1_path = out_dir / "report_next_day_hourly_da.png"
    fig1.savefig(fig1_path, dpi=120)
    plt.close(fig1)

    # next-week average day ahead across simulated paths 
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.hist(weekly_avg_per_path, bins=min(50, max(10, paths.shape[0] // 20)), color="C2", alpha=0.75)
    ax2.axvline(w_mean, color="C0", linewidth=2, label=f"Mean = {w_mean:.2f}")
    ax2.axvline(w_p50, color="C1", linestyle="--", linewidth=1.5, label=f"p50 = {w_p50:.2f}")
    ax2.set_xlabel("Weekly average DA price (EUR/MWh)")
    ax2.set_ylabel("Number of paths")
    ax2.set_title(
        "Distribution of next-week average hourly DA\n"
        "(mean of 168 recursive hourly forecasts per path)"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2_path = out_dir / "report_next_week_avg_da_distribution.png"
    fig2.savefig(fig2_path, dpi=120)
    plt.close(fig2)

    logger.info(
        "Forecast report: wrote CSVs under %s and figures report_next_day_hourly_da.png, "
        "report_next_week_avg_da_distribution.png",
        out_dir,
    )
    return ForecastReportArtifacts(
        next_day_hourly_csv=hourly_csv,
        next_week_avg_summary_csv=week_csv,
        next_week_avg_per_path_csv=per_path_csv,
        combined_summary_csv=combined_csv,
        figure_next_day_da_png=fig1_path,
        figure_next_week_avg_distribution_png=fig2_path,
    )



# Backtest report
def run_backtest_report(
    bundle_dir: Path,
    out_dir: Path,
    *,
    allow_irregular_hourly: bool = True,
    origin_step_hours: int | None = None,
    models_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run walk-forward  backtest on the full data history,
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if models_path is not None:
        artifact = load_production_artifact(Path(models_path))
        cfg = copy.deepcopy(artifact.config)
    else:
        cfg = ModelPipelineConfig()

    cfg.outputs.base_dir = out_dir
    if allow_irregular_hourly:
        cfg.validation.require_strictly_hourly = False
    if origin_step_hours is not None:
        cfg.backtest.origin_step_hours = origin_step_hours

    df = load_smard_bundle_hourly(Path(bundle_dir), cfg)
    logger.info(
        "Backtest frame: %d rows, %s → %s",
        len(df), df.index.min(), df.index.max(),
    )

    result = run_rolling_backtest(df, cfg, output_dir=out_dir)
    oof: pd.DataFrame = result["oof_predictions"]
    fold_df: pd.DataFrame = result["fold_metrics"]

    n_folds = len(fold_df)
    logger.info("Walk-forward completed: %d folds", n_folds)

    summary = _build_summary(oof, fold_df, cfg)
    summary_csv = out_dir / "backtest_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    _plot_oof_timeseries(oof, cfg, out_dir)
    _plot_fold_mae_over_time(fold_df, out_dir)
    _plot_actual_vs_predicted(oof, out_dir)
    _plot_residual_distribution(oof, out_dir)

    logger.info("Backtest report written to %s", out_dir)
    return {**result, "summary": summary}


def _build_summary(
    oof: pd.DataFrame, fold_df: pd.DataFrame, cfg: ModelPipelineConfig
) -> dict[str, Any]:
    actual = oof["y_actual"].values
    point = oof["y_point"].values
    err = actual - point
    q10 = oof["q10"].values
    q90 = oof["q90"].values

    coverage = float(np.mean((actual >= q10) & (actual <= q90)))

    neg_mask = actual < 0
    mae_neg = float(np.mean(np.abs(err[neg_mask]))) if neg_mask.any() else float("nan")
    n_neg = int(neg_mask.sum())

    abs_a = np.abs(actual)
    spike_thr = np.percentile(abs_a, 90) if len(abs_a) else 0.0
    spike_m = abs_a >= spike_thr
    mae_spike = float(np.mean(np.abs(err[spike_m]))) if spike_m.any() else float("nan")

    return {
        "n_oof_hours": len(oof),
        "n_folds": len(fold_df),
        "oof_start": str(oof.index.min()),
        "oof_end": str(oof.index.max()),
        "overall_mae_eur_mwh": float(np.mean(np.abs(err))),
        "overall_rmse_eur_mwh": float(np.sqrt(np.mean(err**2))),
        "overall_median_ae_eur_mwh": float(np.median(np.abs(err))),
        "overall_mean_error_eur_mwh": float(np.mean(err)),
        "q10_q90_coverage": coverage,
        "mae_negative_price_hours": mae_neg,
        "n_negative_price_hours": n_neg,
        "mae_top_decile_abs_price": mae_spike,
        "fold_mae_mean": float(fold_df["mae"].mean()),
        "fold_mae_std": float(fold_df["mae"].std()),
        "fold_mae_min": float(fold_df["mae"].min()),
        "fold_mae_max": float(fold_df["mae"].max()),
        "window_type": cfg.backtest.window_type,
        "origin_step_hours": cfg.backtest.origin_step_hours,
        "validation_block_hours": cfg.backtest.validation_block_hours,
        "gap_hours": cfg.backtest.gap_hours,
    }


def _plot_oof_timeseries(oof: pd.DataFrame, cfg: ModelPipelineConfig, out_dir: Path) -> None:
    tz = cfg.validation.delivery_timezone
    fig, ax = plt.subplots(figsize=(14, 5))
    loc_idx = oof.index.tz_convert(tz)
    ax.plot(loc_idx, oof["y_actual"], linewidth=0.5, alpha=0.7, label="Actual DA", color="black")
    ax.plot(loc_idx, oof["y_point"], linewidth=0.5, alpha=0.7, label="OOF point forecast", color="C0")
    ax.fill_between(
        loc_idx, oof["q10"], oof["q90"],
        alpha=0.15, color="C0", label="q10–q90 band",
    )
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Walk-forward out-of-fold: actual vs point forecast")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m", tz=tz))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "backtest_oof_timeseries.png", dpi=120)
    plt.close(fig)


def _plot_fold_mae_over_time(fold_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(fold_df)), fold_df["mae"], color="C1", alpha=0.8)
    mean_mae = fold_df["mae"].mean()
    ax.axhline(mean_mae, color="C3", linestyle="--", linewidth=1.5, label=f"Mean MAE = {mean_mae:.2f}")
    ax.set_xlabel("Fold index (chronological)")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Walk-forward MAE per fold (time-ordered)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "backtest_fold_mae.png", dpi=120)
    plt.close(fig)


def _plot_actual_vs_predicted(oof: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(oof["y_actual"], oof["y_point"], s=1, alpha=0.3, color="C0")
    lo = min(oof["y_actual"].min(), oof["y_point"].min())
    hi = max(oof["y_actual"].max(), oof["y_point"].max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", linewidth=0.8, label="Perfect")
    ax.set_xlabel("Actual (EUR/MWh)")
    ax.set_ylabel("Predicted (EUR/MWh)")
    ax.set_title("OOF actual vs predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_dir / "backtest_actual_vs_predicted.png", dpi=120)
    plt.close(fig)


def _plot_residual_distribution(oof: pd.DataFrame, out_dir: Path) -> None:
    residual = oof["y_actual"] - oof["y_point"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.hist(residual, bins=80, color="C2", alpha=0.75, edgecolor="white", linewidth=0.3)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.axvline(residual.mean(), color="C3", linestyle="--", label=f"Mean = {residual.mean():.2f}")
    ax1.set_xlabel("Residual (EUR/MWh)")
    ax1.set_ylabel("Count")
    ax1.set_title("Residual distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    tz = "Europe/Berlin"
    loc = oof.index.tz_convert(tz)
    hours = loc.hour
    hourly_mae = pd.Series(np.abs(residual.values), index=hours).groupby(level=0).mean()
    ax2.bar(hourly_mae.index, hourly_mae.values, color="C4", alpha=0.8)
    ax2.set_xlabel("Hour of day (local)")
    ax2.set_ylabel("MAE (EUR/MWh)")
    ax2.set_title("MAE by hour of day")
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "backtest_residuals.png", dpi=120)
    plt.close(fig)
