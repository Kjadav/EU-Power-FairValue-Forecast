"""Run walk-forward regression backtest and build presentation artifacts (tables + figure)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.regression.synthetic_bundle import write_synthetic_regression_bundle
from backtest.regression.walkforward import walk_forward_day_ahead_backtest


def repo_root() -> Path:
    """Repository root (parent of ``backtest/``)."""
    return Path(__file__).resolve().parents[2]


def resolve_smard_bundle_dir(root: Path) -> Path:
    """Prefer ``data/processed/smard_bundle``, then ``smard_bundle_qa_run``."""
    primary = root / "data" / "processed" / "smard_bundle"
    fallback = root / "data" / "processed" / "smard_bundle_qa_run"
    if (primary / "day_ahead_prices.parquet").is_file():
        return primary
    if (fallback / "day_ahead_prices.parquet").is_file():
        return fallback
    return primary


def default_bundle_dir() -> Path:
    return resolve_smard_bundle_dir(repo_root())


def backtest_result_to_overall_df(result: dict[str, Any]) -> pd.DataFrame:
    om = result["overall_metrics"]
    df = pd.DataFrame(om).T.rename(columns={"n": "n_hours", "mae": "MAE", "rmse": "RMSE"}).round(4)
    df.index.name = "model"
    return df


def backtest_result_to_per_day_df(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in result["per_day"]:
        r: dict[str, Any] = {
            "eval_date_utc": row["eval_date_utc"],
            "n_train_hours": row["n_train_hours"],
        }
        for model in ("regression", "naive_last", "naive_seasonal_24h"):
            m = row[model]
            r[f"{model}_mae"] = m["mae"]
            r[f"{model}_rmse"] = m["rmse"]
            r[f"{model}_n"] = m["n"]
        rows.append(r)
    return pd.DataFrame(rows).round(4)


def backtest_result_to_figure(result: dict[str, Any]):
    import matplotlib.pyplot as plt
    import numpy as np

    om = result["overall_metrics"]
    models = list(om.keys())
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x, [om[m]["mae"] for m in models])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=25, ha="right")
    axes[0].set_ylabel("MAE (EUR/MWh)")
    axes[0].set_title("Mean absolute error")
    axes[1].bar(x, [om[m]["rmse"] for m in models])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=25, ha="right")
    axes[1].set_ylabel("RMSE (EUR/MWh)")
    axes[1].set_title("Root mean squared error")
    fig.suptitle("Walk-forward day-ahead hourly backtest")
    fig.tight_layout()
    return fig


def print_regression_backtest_report(
    result: dict[str, Any],
    *,
    fallback_note: str | None = None,
) -> None:
    """Stdout summary for scripts / logging (same lines as former CLI)."""
    o = result["overall_metrics"]
    print("Walk-forward day-ahead hourly backtest (UTC)")
    print(f"  bundle: {result['bundle_dir']}")
    if fallback_note:
        print(f"  note (synthetic fallback was used): {fallback_note}")
    print(f"  eval days completed: {result['n_eval_days_completed']} (requested {result['n_eval_days_requested']})")
    print(f"  forecast window: {result['forecast_index_start']} .. {result['forecast_index_end']}")
    print(f"  model: {result['spec']}")
    print()
    for name in ("regression", "naive_last", "naive_seasonal_24h"):
        m = o[name]
        print(f"  {name:22s}  n={m['n']:4d}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")


def deploy_regression_backtest(
    bundle_dir: Path | None = None,
    *,
    n_eval_days: int = 7,
    min_train_hours: int = 200,
    use_synthetic_fallback: bool = False,
    synthetic_eval_days_cap: int = 3,
    synthetic_min_train: int = 150,
    json_path: Path | None = None,
    print_summary: bool = True,
) -> dict[str, Any]:
    """
    Single entry for production / notebooks / scripts: run backtest, build tables + figure,
    optionally print and write JSON.

    Returns a dict with ``result`` (raw walk-forward dict), ``overall_df``, ``per_day_df``,
    ``figure``, ``bundle_dir_used``, and ``fallback_note`` (if synthetic path was used).
    """
    b = bundle_dir if bundle_dir is not None else default_bundle_dir()
    fallback_note: str | None = None
    bundle_used = b

    try:
        raw = walk_forward_day_ahead_backtest(
            b,
            n_eval_days=n_eval_days,
            min_train_hours=min_train_hours,
        )
    except ValueError as e:
        if not use_synthetic_fallback:
            raise
        fallback_note = str(e)
        tmp = Path(tempfile.mkdtemp(prefix="regression_backtest_"))
        write_synthetic_regression_bundle(tmp)
        bundle_used = tmp
        raw = walk_forward_day_ahead_backtest(
            tmp,
            n_eval_days=min(n_eval_days, synthetic_eval_days_cap),
            min_train_hours=synthetic_min_train,
        )

    overall_df = backtest_result_to_overall_df(raw)
    per_day_df = backtest_result_to_per_day_df(raw)
    fig = backtest_result_to_figure(raw)

    if print_summary:
        print_regression_backtest_report(raw, fallback_note=fallback_note)

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        raw_for_json = {k: v for k, v in raw.items() if k != "forecast_frame"}
        json_path.write_text(json.dumps(raw_for_json, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {json_path.resolve()}")

    return {
        "result": raw,
        "forecast_frame": raw["forecast_frame"],
        "overall_df": overall_df,
        "per_day_df": per_day_df,
        "figure": fig,
        "bundle_dir_used": bundle_used.resolve(),
        "fallback_note": fallback_note,
    }


def _style_utc_time_xaxis(ax, index: pd.DatetimeIndex) -> None:
    """Unambiguous axis labels (year-month-day hour) in the index timezone (SMARD = UTC)."""
    import matplotlib.dates as mdates

    tz = index.tz
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=14))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=tz))
    ax.set_xlabel("Time (UTC)")
    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment("center")


def plot_prediction_timeseries(forecast_frame: pd.DataFrame):
    """Line chart: actual vs all models over the walk-forward forecast window."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for col in forecast_frame.columns:
        ax.plot(forecast_frame.index, forecast_frame[col], label=col, linewidth=1.2, alpha=0.9)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Day-ahead price: actual vs walk-forward forecasts")
    ax.legend(loc="upper right", fontsize=9)
    _style_utc_time_xaxis(ax, forecast_frame.index)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted_scatter(forecast_frame: pd.DataFrame, *, model_col: str = "regression"):
    """45° reference: predicted vs realized (same units)."""
    import matplotlib.pyplot as plt

    a = forecast_frame["actual"]
    p = forecast_frame[model_col]
    lo = float(min(a.min(), p.min()))
    hi = float(max(a.max(), p.max()))
    pad = (hi - lo) * 0.05 + 1.0
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(a, p, alpha=0.45, s=22, edgecolors="none")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Actual (EUR/MWh)")
    ax.set_ylabel(f"Predicted — {model_col} (EUR/MWh)")
    ax.set_title("Realized vs forecast (walk-forward hours)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_forecast_error_timeseries(forecast_frame: pd.DataFrame):
    """Per-model forecast error (actual − predicted) over time."""
    import matplotlib.pyplot as plt

    actual = forecast_frame["actual"]
    model_cols = [c for c in forecast_frame.columns if c != "actual"]
    fig, axes = plt.subplots(len(model_cols), 1, figsize=(12, 2.2 * len(model_cols)), sharex=True)
    if len(model_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, model_cols, strict=True):
        err = actual - forecast_frame[col]
        ax.fill_between(forecast_frame.index, err, 0, alpha=0.35)
        ax.plot(forecast_frame.index, err, linewidth=0.9, color="C0")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel(f"actual −\n{col}")
    _style_utc_time_xaxis(axes[-1], forecast_frame.index)
    fig.suptitle("Walk-forward forecast errors (EUR/MWh)")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
