"""LightGBM / XGBoost training, sample weights, hyperparameter search."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

from trading_pipeline_utils.settings import ModelPipelineConfig

logger = logging.getLogger(__name__)


def compute_sample_weights(
    price: pd.Series,
    train_mask: np.ndarray,
    timestamps: pd.DatetimeIndex,
    *,
    neg_price_extra: float,
    half_life_days: float,
) -> np.ndarray:
    """Train-only spike thresholds; recency decay from last train timestamp."""
    p = price.values.astype(np.float64)
    w = np.ones(len(price), dtype=np.float64)
    tm = np.flatnonzero(train_mask)
    if tm.size == 0:
        return w
    tr = p[train_mask]
    tr = tr[np.isfinite(tr)]
    abs_p95 = float(np.percentile(np.abs(tr), 95))
    upper_spike = float(np.percentile(tr, 97.5))
    lower_spike = float(np.percentile(tr, 2.5))
    last_ts = timestamps[train_mask].max()
    delta = last_ts - timestamps
    age_days = np.asarray(delta.total_seconds(), dtype=np.float64) / 86400.0
    recency = np.exp(-np.log(2.0) * np.maximum(age_days, 0.0) / half_life_days)
    spike_w = 1.0 + np.clip(np.abs(p) / max(abs_p95, 1e-9), 0.0, 3.0)
    spike_w = np.where((p > upper_spike) | (p < lower_spike), spike_w * 2.0, spike_w)
    spike_w = np.where(p < 0, spike_w * neg_price_extra, spike_w)
    w = spike_w * recency
    w[~np.isfinite(w)] = 1.0
    return w


@dataclass
class TrainedModelBundle:
    lgbm_point: Any
    lgbm_q10: Any
    lgbm_q50: Any
    lgbm_q90: Any
    xgb_point: Any | None
    xgb_q10: Any | None
    xgb_q50: Any | None
    xgb_q90: Any | None
    best_params: dict[str, Any]
    categorical_feature: list[str]


def _lgbm_point_params(base: dict[str, Any]) -> dict[str, Any]:
    p = dict(base)
    p["objective"] = "huber"
    p["alpha"] = 0.9
    return p


def _lgbm_quantile_params(base: dict[str, Any], alpha: float) -> dict[str, Any]:
    p = dict(base)
    p["objective"] = "quantile"
    p["alpha"] = alpha
    return p


def _sample_lgbm_grid(cfg: Any, rng: np.random.Generator) -> list[dict[str, Any]]:
    hs = cfg.hyperparam_search
    space = list(
        product(
            hs.num_leaves_choices,
            hs.max_depth_choices,
            hs.min_data_in_leaf_choices,
            hs.feature_fraction_choices,
            hs.bagging_fraction_choices,
            hs.lambda_l1_choices,
            hs.lambda_l2_choices,
            hs.min_gain_to_split_choices,
        )
    )
    if len(space) <= hs.max_trials:
        chosen = space
    else:
        idx = rng.choice(len(space), size=hs.max_trials, replace=False)
        chosen = [space[i] for i in idx]
    out = []
    for tup in chosen:
        out.append(
            {
                "num_leaves": tup[0],
                "max_depth": tup[1],
                "min_data_in_leaf": tup[2],
                "feature_fraction": tup[3],
                "bagging_fraction": tup[4],
                "lambda_l1": tup[5],
                "lambda_l2": tup[6],
                "min_gain_to_split": tup[7],
            }
        )
    return out


def train_lightgbm_suite(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    sw_train: np.ndarray,
    categorical_feature: list[str],
    config: Any,
) -> TrainedModelBundle:
    fixed = config.lgbm
    base_kwargs: dict[str, Any] = {
        "boosting_type": fixed.boosting_type,
        "learning_rate": fixed.learning_rate,
        "n_estimators": fixed.n_estimators,
        "max_bin": fixed.max_bin,
        "random_state": fixed.random_state,
        "n_jobs": fixed.n_jobs,
        "verbose": fixed.verbose,
        "force_col_wise": True,
    }
    if "deterministic" in inspect.signature(LGBMRegressor.__init__).parameters:
        base_kwargs["deterministic"] = True

    cbs = [
        early_stopping(stopping_rounds=fixed.early_stopping_rounds, verbose=False),
        log_evaluation(period=0),
    ]

    rng = np.random.default_rng(config.hyperparam_search.random_state)
    best_params: dict[str, Any] = {}
    best_mae = float("inf")

    if config.hyperparam_search.enabled:
        for trial in _sample_lgbm_grid(config, rng):
            params_pt = _lgbm_point_params({**base_kwargs, **trial})
            m = LGBMRegressor(**params_pt)
            m.fit(
                X_train,
                y_train,
                sample_weight=sw_train,
                eval_set=[(X_val, y_val)],
                eval_metric="mae",
                categorical_feature=categorical_feature,
                callbacks=cbs,
            )
            pred = m.predict(X_val)
            mae = float(np.nanmean(np.abs(y_val - pred)))
            if mae < best_mae:
                best_mae = mae
                best_params = trial
        logger.info("Hyperparam search best val MAE=%.5f params=%s", best_mae, best_params)
    else:
        best_params = {
            "num_leaves": fixed.num_leaves,
            "max_depth": fixed.max_depth,
            "min_data_in_leaf": fixed.min_data_in_leaf,
            "feature_fraction": fixed.feature_fraction,
            "bagging_fraction": fixed.bagging_fraction,
            "lambda_l1": fixed.lambda_l1,
            "lambda_l2": fixed.lambda_l2,
            "min_gain_to_split": fixed.min_gain_to_split,
        }

    final_base = {**base_kwargs, **best_params}

    pt = LGBMRegressor(**_lgbm_point_params(final_base))
    pt.fit(
        X_train,
        y_train,
        sample_weight=sw_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        categorical_feature=categorical_feature,
        callbacks=cbs,
    )

    def _fit_q(alpha: float) -> Any:
        m = LGBMRegressor(**_lgbm_quantile_params(final_base, alpha))
        m.fit(
            X_train,
            y_train,
            sample_weight=sw_train,
            eval_set=[(X_val, y_val)],
            eval_metric="quantile",
            categorical_feature=categorical_feature,
            callbacks=cbs,
        )
        return m

    q10, q50, q90 = _fit_q(0.1), _fit_q(0.5), _fit_q(0.9)

    xgb_p = xgb_q10 = xgb_q50 = xgb_q90 = None
    if config.run_xgboost_benchmark:
        if xgb is None:
            logger.warning("XGBoost benchmark skipped: xgboost not installed")
        else:
            try:
                xgb_p = xgb.XGBRegressor(
                    objective="reg:pseudohubererror",
                    n_estimators=2000,
                    learning_rate=0.03,
                    max_depth=8,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=fixed.random_state,
                    early_stopping_rounds=fixed.early_stopping_rounds,
                )
                xgb_p.fit(
                    X_train, y_train, sample_weight=sw_train, eval_set=[(X_val, y_val)], verbose=False
                )

                def _xgb_q(q: float) -> Any:
                    m = xgb.XGBRegressor(
                        objective="reg:quantileerror",
                        quantile_alpha=q,
                        n_estimators=2000,
                        learning_rate=0.03,
                        max_depth=8,
                        random_state=fixed.random_state,
                        early_stopping_rounds=fixed.early_stopping_rounds,
                    )
                    m.fit(
                        X_train, y_train, sample_weight=sw_train, eval_set=[(X_val, y_val)], verbose=False
                    )
                    return m

                xgb_q10, xgb_q50, xgb_q90 = _xgb_q(0.1), _xgb_q(0.5), _xgb_q(0.9)
            except Exception as e:  # noqa: BLE001
                logger.warning("XGBoost benchmark skipped: %s", e)

    return TrainedModelBundle(
        lgbm_point=pt,
        lgbm_q10=q10,
        lgbm_q50=q50,
        lgbm_q90=q90,
        xgb_point=xgb_p,
        xgb_q10=xgb_q10,
        xgb_q50=xgb_q50,
        xgb_q90=xgb_q90,
        best_params=best_params,
        categorical_feature=categorical_feature,
    )
