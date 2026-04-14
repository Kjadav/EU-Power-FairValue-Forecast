from __future__ import annotations

import copy
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from trading_pipeline_utils.features.engineering import build_feature_matrix
from trading_pipeline_utils.features.transforms import ArcsinhPriceTransform
from trading_pipeline_utils.forecasting.loader import load_smard_bundle_hourly
from trading_pipeline_utils.forecasting.metrics import enforce_quantile_monotonicity
from trading_pipeline_utils.forecasting.simulation import (
    compute_standardized_residuals,
    encode_bucket,
    simulate_paths_next_week,
)
from trading_pipeline_utils.models.lgbm_compat import safe_lgbm_predict
from trading_pipeline_utils.models.training import (
    TrainedModelBundle,
    compute_sample_weights,
    train_lightgbm_suite,
)
from trading_pipeline_utils.settings import ColumnMap, ModelPipelineConfig
from trading_pipeline_utils.validation.forecast_validation import validate_hourly_frame

logger = logging.getLogger(__name__)


@dataclass
class ProductionModelArtifact:
    trained: TrainedModelBundle
    target_transform: ArcsinhPriceTransform
    train_abs_price_p95: float
    z_train: np.ndarray
    train_bucket_ids: np.ndarray
    config: ModelPipelineConfig


def save_production_artifact(artifact: ProductionModelArtifact, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    logger.info("Wrote production artifact %s", path)


def load_production_artifact(path: Path) -> ProductionModelArtifact:
    # Older joblib pickles may reference ``cobblestone.forecasting.*``; map those to this package.
    import trading_pipeline_utils.models.inference as _inference
    import trading_pipeline_utils.models.training as _training
    import trading_pipeline_utils.features.transforms as _transforms
    import trading_pipeline_utils.settings as _settings

    if "cobblestone" not in sys.modules:
        root = types.ModuleType("cobblestone")
        root.__path__ = []
        sys.modules["cobblestone"] = root
    if "cobblestone.forecasting" not in sys.modules:
        fcst = types.ModuleType("cobblestone.forecasting")
        fcst.__path__ = []
        sys.modules["cobblestone.forecasting"] = fcst
        sys.modules["cobblestone"].forecasting = fcst

    for old_name, new_module in (
        ("cobblestone.forecasting.production", _inference),
        ("cobblestone.forecasting.model_training", _training),
        ("cobblestone.forecasting.target_transform", _transforms),
        ("cobblestone.forecasting.config", _settings),
    ):
        if old_name not in sys.modules:
            sys.modules[old_name] = new_module
    _settings_mod = sys.modules.get("cobblestone.forecasting.config")
    if _settings_mod is not None and not hasattr(_settings_mod, "ModelPipelineConfig"):
        _settings_mod.ModelPipelineConfig = ModelPipelineConfig
        _settings_mod.ColumnMap = ColumnMap

    obj = joblib.load(path)
    if not isinstance(obj, ProductionModelArtifact):
        raise TypeError(f"Expected ProductionModelArtifact in {path}, got {type(obj)}")
    return obj


def fit_production_model(df: pd.DataFrame, config: ModelPipelineConfig) -> ProductionModelArtifact:
    c = config.columns
    n = len(df)
    price_vec = df[c.price_da].astype(np.float64).values
    price_tr = df[c.price_da].dropna()
    if len(price_tr) < 50:
        raise ValueError("Need at least 50 finite prices to fit production model")
    p95 = float(np.percentile(np.abs(price_tr.values), 95))

    X, _num_cols, cat_cols = build_feature_matrix(df, config, train_abs_price_p95=p95)
    row_ok = (~X.isna().all(axis=1).values) & np.isfinite(price_vec)

    tr_pos = np.flatnonzero(row_ok)
    if len(tr_pos) < 120:
        raise ValueError("Too few complete rows for production training")

    tfm = ArcsinhPriceTransform(scale_floor=config.target.scale_floor)
    tfm.fit(pd.Series(price_vec[tr_pos]))
    y_t = np.full(n, np.nan, dtype=np.float64)
    y_t[tr_pos] = tfm.transform(price_vec[tr_pos])

    tr_row_mask = np.zeros(n, dtype=bool)
    tr_row_mask[tr_pos] = True
    sw = compute_sample_weights(
        df[c.price_da],
        tr_row_mask,
        df.index,
        neg_price_extra=config.sample_weight.neg_price_extra,
        half_life_days=config.sample_weight.half_life_days,
    )

    split_at = max(len(tr_pos) - 28 * 24, int(len(tr_pos) * 0.88))
    inner_fit = np.zeros(n, dtype=bool)
    inner_es = np.zeros(n, dtype=bool)
    inner_fit[tr_pos[:split_at]] = True
    inner_es[tr_pos[split_at:]] = True

    fit_ix = np.flatnonzero(inner_fit)
    es_ix = np.flatnonzero(inner_es)
    bundle = train_lightgbm_suite(
        X.iloc[fit_ix],
        y_t[fit_ix],
        X.iloc[es_ix],
        y_t[es_ix],
        sw[inner_fit],
        cat_cols,
        config,
    )

    X_tr = X.iloc[tr_pos]
    pred_t = safe_lgbm_predict(bundle.lgbm_point, X_tr)
    q10_t = safe_lgbm_predict(bundle.lgbm_q10, X_tr)
    q90_t = safe_lgbm_predict(bundle.lgbm_q90, X_tr)
    y_actual = price_vec[tr_pos]
    z = compute_standardized_residuals(
        y_actual,
        pred_t,
        q10_t,
        q90_t,
        tfm,
        config.simulation.sigma_floor,
    )

    loc = df.index[tr_pos].tz_convert(config.validation.delivery_timezone)
    hour_local = np.asarray(loc.hour, dtype=np.int64)
    weekend = np.asarray((loc.dayofweek >= 5).astype(np.int64), dtype=np.int64)
    month = np.asarray(loc.month, dtype=np.int64)
    train_bucket_ids = encode_bucket(hour_local, weekend, month)

    fin = np.isfinite(z)
    z_train = z[fin]
    train_bucket_ids = train_bucket_ids[fin]

    cfg_copy = copy.deepcopy(config)
    return ProductionModelArtifact(
        trained=bundle,
        target_transform=tfm,
        train_abs_price_p95=p95,
        z_train=z_train,
        train_bucket_ids=train_bucket_ids,
        config=cfg_copy,
    )


def predict_recursive_hours(
    df_hist: pd.DataFrame,
    config: ModelPipelineConfig,
    artifact: ProductionModelArtifact,
    n_hours: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if n_hours < 1:
        raise ValueError("n_hours must be >= 1")
    c = config.columns
    bundle = artifact.trained
    tfm = artifact.target_transform
    p95 = artifact.train_abs_price_p95

    dfw = df_hist.copy()
    last_ts = dfw.index.max()
    fut = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=n_hours, freq="h", tz="UTC")

    yhat_list: list[float] = []
    q10_raw: list[float] = []
    q90_raw: list[float] = []
    rows_out: list[dict[str, Any]] = []

    for ts in fut:
        row: pd.Series | None = None
        for delta in (pd.Timedelta(hours=24), pd.Timedelta(days=7)):
            cand = ts - delta
            if cand in df_hist.index:
                row = df_hist.loc[cand].copy()
                break
        if row is None:
            sub = df_hist.loc[: ts - pd.Timedelta(hours=1)]
            if len(sub) == 0:
                raise ValueError(f"No history before {ts} to build future row")
            row = sub.iloc[-1].copy()
        row[c.price_da] = np.nan
        chunk = pd.DataFrame([row.to_dict()], index=[ts])
        dfw = pd.concat([dfw, chunk])
        dfw.index = pd.to_datetime(dfw.index, utc=True)
        dfw = dfw.sort_index()
        dfw = validate_hourly_frame(dfw, config)

        X, _n, _cat = build_feature_matrix(dfw, config, train_abs_price_p95=p95)
        x1 = X.loc[[ts]]
        if bool(x1.isna().all(axis=1).iloc[0]):
            raise ValueError(f"All-NaN feature row at {ts}")

        pt = float(safe_lgbm_predict(bundle.lgbm_point, x1)[0])
        q10 = float(safe_lgbm_predict(bundle.lgbm_q10, x1)[0])
        q50 = float(safe_lgbm_predict(bundle.lgbm_q50, x1)[0])
        q90 = float(safe_lgbm_predict(bundle.lgbm_q90, x1)[0])
        yhat_list.append(pt)
        q10_raw.append(q10)
        q90_raw.append(q90)

        p10_eur = float(tfm.inverse_transform(np.array([q10]))[0])
        p50_eur = float(tfm.inverse_transform(np.array([q50]))[0])
        p90_eur = float(tfm.inverse_transform(np.array([q90]))[0])
        p10_eur, p50_eur, p90_eur = enforce_quantile_monotonicity(
            np.array([p10_eur]),
            np.array([p50_eur]),
            np.array([p90_eur]),
        )
        price_point = float(tfm.inverse_transform(np.array([pt]))[0])
        dfw.loc[ts, c.price_da] = price_point

        rows_out.append(
            {
                "y_point": price_point,
                "q10": float(p10_eur[0]),
                "q50": float(p50_eur[0]),
                "q90": float(p90_eur[0]),
            }
        )

    fc = pd.DataFrame(rows_out, index=fut)
    yhat_t = np.asarray(yhat_list, dtype=np.float64)
    q10_t = np.asarray(q10_raw, dtype=np.float64)
    q90_t = np.asarray(q90_raw, dtype=np.float64)
    return fc, yhat_t, q10_t, q90_t


def run_forecast_next_day(
    bundle_dir: Path,
    models_path: Path,
    out_dir: Path | None = None,
    *,
    allow_irregular_hourly: bool = False,
) -> pd.DataFrame:
    artifact = load_production_artifact(Path(models_path))
    cfg = copy.deepcopy(artifact.config)
    if allow_irregular_hourly:
        cfg.validation.require_strictly_hourly = False
    if out_dir is not None:
        cfg.outputs.base_dir = Path(out_dir)

    df = load_smard_bundle_hourly(Path(bundle_dir), cfg)
    fc, _yt, _q10, _q90 = predict_recursive_hours(df, cfg, artifact, n_hours=24)

    out_dir_p = Path(cfg.outputs.base_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out_dir_p / cfg.outputs.final_forecasts_parquet)
    logger.info("Wrote %s", out_dir_p / cfg.outputs.final_forecasts_parquet)
    return fc


def run_simulate_next_week(
    bundle_dir: Path,
    models_path: Path,
    out_dir: Path | None = None,
    *,
    allow_irregular_hourly: bool = False,
) -> dict[str, Any]:
    artifact = load_production_artifact(Path(models_path))
    cfg = copy.deepcopy(artifact.config)
    if allow_irregular_hourly:
        cfg.validation.require_strictly_hourly = False
    if out_dir is not None:
        cfg.outputs.base_dir = Path(out_dir)

    df = load_smard_bundle_hourly(Path(bundle_dir), cfg)
    H = int(cfg.simulation.horizon_hours)
    fc, yhat_t, q10_t, q90_t = predict_recursive_hours(df, cfg, artifact, n_hours=H)
    fut = fc.index

    loc = fut.tz_convert(cfg.validation.delivery_timezone)
    hour_local = np.asarray(loc.hour, dtype=np.int64)
    weekend = np.asarray((loc.dayofweek >= 5).astype(np.int64), dtype=np.int64)
    month = np.asarray(loc.month, dtype=np.int64)
    future_bucket_ids = encode_bucket(hour_local, weekend, month)

    sim_out = simulate_paths_next_week(
        yhat_t,
        q10_t,
        q90_t,
        artifact.target_transform,
        artifact.z_train,
        future_bucket_ids,
        artifact.train_bucket_ids,
        cfg,
        future_index=fut,
    )

    out_dir_p = Path(cfg.outputs.base_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    paths = sim_out["paths"]
    sim_df = pd.DataFrame(
        paths.T,
        index=fut,
        columns=[f"path_{i}" for i in range(paths.shape[0])],
    )
    sim_df.to_parquet(out_dir_p / cfg.outputs.simulated_paths_parquet)

    summary_row = {k: v for k, v in sim_out.items() if k != "paths"}
    pd.DataFrame([summary_row]).to_csv(out_dir_p / cfg.outputs.weekly_distributions_csv, index=False)
    logger.info(
        "Wrote %s and %s",
        out_dir_p / cfg.outputs.simulated_paths_parquet,
        out_dir_p / cfg.outputs.weekly_distributions_csv,
    )
    return sim_out
