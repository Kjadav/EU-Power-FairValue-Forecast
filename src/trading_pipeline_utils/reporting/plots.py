
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_pipeline_utils.market.adapter import ForecastSnapshot
from trading_pipeline_utils.market.curve import TranslatedProduct
from trading_pipeline_utils.market.signals import Signal

logger = logging.getLogger(__name__)


def plot_hourly_forecast(
    bundle: ForecastSnapshot,
    out_dir: Path,
    *,
    actual_prices: pd.Series | None = None,
) -> Path:
    """Figure 1: Day-ahead hourly forecast fan chart with distribution bands."""
    fc = bundle.point_forecasts
    tz = bundle.delivery_timezone
    loc = fc.index.tz_convert(tz)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(fc))

    ax.fill_between(
        x, fc["price_p10"].values, fc["price_p90"].values,
        alpha=0.25, color="C0", label="p10–p90",
    )
    ax.plot(x, fc["price_p50"].values, ":", color="C1", label="Median (p50)")
    ax.plot(x, fc["price_mean"].values, "-", color="C0", linewidth=2, label="Point forecast")

    if actual_prices is not None:
        common = fc.index.intersection(actual_prices.index)
        if len(common) > 0:
            idx_pos = [int(np.searchsorted(fc.index, t)) for t in common if t in fc.index]
            ax.scatter(idx_pos, actual_prices.loc[common].values, color="black", s=15, zorder=5, label="Actual")

    next_day_avg = float(fc["price_mean"].mean())
    ax.axhline(next_day_avg, color="C3", linestyle="--", alpha=0.6, linewidth=1)
    ax.annotate(
        f"Next-day avg = {next_day_avg:.1f}",
        xy=(len(fc) - 1, next_day_avg), fontsize=8, color="C3",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{loc[i].strftime('%m-%d %H')}h" for i in x], rotation=45, ha="right")
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"Day-ahead hourly price forecast ({tz})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out_dir / "da_hourly_forecast.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    csv_path = out_dir / "da_hourly_forecast.csv"
    fc.to_csv(csv_path)
    return path


def plot_weekly_distribution(
    bundle: ForecastSnapshot,
    signals: dict[str, Signal],
    out_dir: Path,
) -> Path:
    """Figure 2: Next-week distribution histogram with confidence-weighted signal."""
    fig, ax = plt.subplots(figsize=(9, 5))

    if bundle.scenario_paths is not None:
        paths_df = bundle.scenario_paths
        path_cols = [c for c in paths_df.columns if c.startswith("path_")]
        paths_matrix = paths_df[path_cols].values
        weekly_avg = paths_matrix.mean(axis=0)

        ax.hist(weekly_avg, bins=min(50, max(10, len(weekly_avg) // 20)),
                color="C2", alpha=0.75, edgecolor="white", linewidth=0.3)

        mean_v = float(np.mean(weekly_avg))
        p10_v = float(np.percentile(weekly_avg, 10))
        p50_v = float(np.percentile(weekly_avg, 50))
        p90_v = float(np.percentile(weekly_avg, 90))

        ax.axvline(mean_v, color="C0", linewidth=2, label=f"Mean = {mean_v:.2f}")
        ax.axvline(p10_v, color="C4", linestyle=":", label=f"p10 = {p10_v:.2f}")
        ax.axvline(p50_v, color="C1", linestyle="--", label=f"p50 = {p50_v:.2f}")
        ax.axvline(p90_v, color="C4", linestyle=":", label=f"p90 = {p90_v:.2f}")

        top_signal = _best_signal(signals)
        if top_signal is not None:
            code, sig = top_signal
            label_text = (
                f"Signal: {sig.direction.upper()} "
                f"(conf={sig.confidence:.2f}, z={sig.edge_z:+.2f})"
            )
            ax.annotate(
                label_text,
                xy=(0.02, 0.95), xycoords="axes fraction",
                fontsize=9, fontweight="bold",
                color="C0" if sig.direction == "long" else "C3" if sig.direction == "short" else "gray",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        csv_data = pd.DataFrame({
            "weekly_avg_da_eur_mwh": weekly_avg,
        })
        csv_data.to_csv(out_dir / "weekly_distribution.csv", index=False)
    else:
        ax.text(0.5, 0.5, "No scenario paths available", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Weekly average DA price (EUR/MWh)")
    ax.set_ylabel("Number of paths")
    ax.set_title("Next-week expected average DA distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out_dir / "weekly_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _best_signal(signals: dict[str, Signal]) -> tuple[str, Signal] | None:
    """Return the signal with highest absolute score, preferring prompt_week_base."""
    if not signals:
        return None
    if "prompt_week_base" in signals:
        return "prompt_week_base", signals["prompt_week_base"]
    best_code = max(signals, key=lambda c: abs(signals[c].signal_score))
    return best_code, signals[best_code]


def generate_all_plots(
    bundle: ForecastSnapshot,
    signals: dict[str, Signal],
    out_dir: Path,
    *,
    actual_prices: pd.Series | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    return {
        "hourly_forecast": plot_hourly_forecast(bundle, out_dir, actual_prices=actual_prices),
        "weekly_distribution": plot_weekly_distribution(bundle, signals, out_dir),
    }
