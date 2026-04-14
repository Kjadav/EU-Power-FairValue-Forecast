from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

try:
    import shap
except ImportError:
    shap = None  # type: ignore[assignment]

from trading_pipeline_utils.settings import ModelPipelineConfig

logger = logging.getLogger(__name__)


def export_lightgbm_point_explainability(
    lgbm_point: Any,
    X_sample: pd.DataFrame,
    config: ModelPipelineConfig,
    output_dir: Path,
) -> None:
    """creating feature importance csv to be/if used"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    imp = getattr(lgbm_point, "feature_importances_", None)
    if imp is not None:
        try:
            names = lgbm_point.booster_.feature_name()
        except Exception:  # noqa: BLE001
            names = [str(c) for c in X_sample.columns]
        imp_a = np.asarray(imp, dtype=np.float64).ravel()
        if len(names) != len(imp_a):
            names = [str(c) for c in X_sample.columns][: len(imp_a)]
        pd.DataFrame({"feature": names, "importance": imp_a}).sort_values(
            "importance", ascending=False
        ).to_csv(output_dir / config.outputs.feature_importance_csv, index=False)

    if shap is None:
        logger.warning("shap not installed; skippinp SHAP export")
        return

    n = min(config.explainability.shap_max_samples, len(X_sample))
    if n < 1:
        return
    xs = X_sample.iloc[:n]
    explainer = shap.TreeExplainer(lgbm_point)
    sv = explainer.shap_values(xs)
    if isinstance(sv, list):
        sv = sv[0]
    mean_abs = np.mean(np.abs(np.asarray(sv)), axis=0)
    top_n = config.explainability.plot_top_n
    order = np.argsort(-mean_abs)[:top_n]
    summary = pd.DataFrame(
        {"feature": [X_sample.columns[i] for i in order], "mean_abs_shap": mean_abs[order]}
    )
    summary.to_csv(output_dir / config.outputs.shap_summary_csv, index=False)

    if plt is not None:
        try:
            shap.summary_plot(np.asarray(sv)[:, order], xs.iloc[:, order], show=False)
            plt.tight_layout()
            plt.savefig(output_dir / "shap_summary.png", dpi=120)
            plt.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("SHAP plot skipped: %s", e)
