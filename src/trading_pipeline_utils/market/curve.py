"""
Translation method:
    delta_forward_j_t = alpha_j + beta_j * delta_model_week_t + error_j_t
    Estimated with HuberRegressor (robust to outliers in daily forward changes).
    translated_FV_j = market_forward_j + beta_j * (model_FV_week - market_forward_week)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import HuberRegressor
    _HAS_SKLEARN = True
except ImportError:
    HuberRegressor = None  # type: ignore[assignment, misc]
    _HAS_SKLEARN = False

from trading_pipeline_utils.market.adapter import ForecastSnapshot
from trading_pipeline_utils.settings import PeakDefinition, PostModelConfig

logger = logging.getLogger(__name__)

PRODUCT_CODES = [
    "prompt_week_base",
    "prompt_week_peak",
    "prompt_month_base",
    "prompt_month_peak",
    "prompt_quarter_base",
    "prompt_quarter_peak",
]


@dataclass
class ProductFairValue:
    product_code: str
    strip_type: str  # "base" or "peak"
    fair_value_mean: float
    fair_value_std: float | None = None
    fair_value_p10: float | None = None
    fair_value_p50: float | None = None
    fair_value_p90: float | None = None
    n_hours: int = 0
    n_paths: int = 0
    method: str = "direct_pathwise"


@dataclass
class TranslationDiagnostics:
    beta: float = 0.0
    alpha: float = 0.0
    residual_std: float = 0.0
    rolling_r2: float = 0.0
    beta_stability: float = 0.0
    sign_stability: float = 0.0
    coverage_ratio: float = 0.0
    method_used: str = "huber_regression"
    translation_quality: str = "unknown"
    n_observations: int = 0


@dataclass
class TranslatedProduct:
    product_code: str
    strip_type: str
    market_forward: float
    fair_value: float
    std: float
    p10: float | None
    p50: float | None
    p90: float | None
    edge: float
    risk_premium_proxy: float
    diagnostics: TranslationDiagnostics = field(default_factory=TranslationDiagnostics)
    has_market_forward: bool = True


def _peak_mask_from_index(
    idx: pd.DatetimeIndex,
    tz: str,
    peak_def: PeakDefinition,
) -> np.ndarray:
    loc = idx.tz_convert(tz)
    if peak_def.weekdays_only:
        return np.array(
            [
                (d.dayofweek < 5) and (peak_def.hour_start <= d.hour < peak_def.hour_end)
                for d in loc
            ],
            dtype=bool,
        )
    return np.array(
        [peak_def.hour_start <= d.hour < peak_def.hour_end for d in loc],
        dtype=bool,
    )


def _offpeak_mask(peak: np.ndarray) -> np.ndarray:
    return ~peak


def compute_product_fair_values_pathwise(
    bundle: ForecastSnapshot,
    config: PostModelConfig,
) -> dict[str, ProductFairValue]:
    if bundle.scenario_paths is None:
        raise ValueError(
            "Scenario paths required for pathwise product fair value aggregation. "
            "Run simulate-next-week first."
        )

    paths_df = bundle.scenario_paths
    idx = paths_df.index
    tz = config.products.delivery_timezone
    peak_mask = _peak_mask_from_index(idx, tz, config.products.peak)

    path_cols = [c for c in paths_df.columns if c.startswith("path_")]
    paths_matrix = paths_df[path_cols].values  # (H, n_paths)
    n_hours, n_paths = paths_matrix.shape

    results: dict[str, ProductFairValue] = {}

    for code in ["prompt_week_base", "prompt_week_peak"]:
        strip = "peak" if "peak" in code else "base"
        mask = peak_mask if strip == "peak" else np.ones(n_hours, dtype=bool)
        if not mask.any():
            continue

        fv_per_path = paths_matrix[mask, :].mean(axis=0)  # (n_paths,)
        results[code] = ProductFairValue(
            product_code=code,
            strip_type=strip,
            fair_value_mean=float(np.mean(fv_per_path)),
            fair_value_std=float(np.std(fv_per_path)),
            fair_value_p10=float(np.percentile(fv_per_path, 10)),
            fair_value_p50=float(np.percentile(fv_per_path, 50)),
            fair_value_p90=float(np.percentile(fv_per_path, 90)),
            n_hours=int(mask.sum()),
            n_paths=n_paths,
            method="direct_pathwise",
        )

    return results


def compute_product_fair_values_point(
    bundle: ForecastSnapshot,
    config: PostModelConfig,
) -> dict[str, ProductFairValue]:
    """Deterministic product FVs from point forecasts only (no distribution)."""
    fc = bundle.point_forecasts
    idx = fc.index
    tz = config.products.delivery_timezone
    peak_mask = _peak_mask_from_index(idx, tz, config.products.peak)

    prices = fc["price_mean"].values
    results: dict[str, ProductFairValue] = {}

    for code, strip, mask in [
        ("next_day_base", "base", np.ones(len(prices), dtype=bool)),
        ("next_day_peak", "peak", peak_mask),
    ]:
        if not mask.any():
            continue
        results[code] = ProductFairValue(
            product_code=code,
            strip_type=strip,
            fair_value_mean=float(np.mean(prices[mask])),
            n_hours=int(mask.sum()),
            method="point_mean",
        )

    return results


def _fit_translation_regression(
    delta_week: np.ndarray,
    delta_product: np.ndarray,
    epsilon: float = 1.35,
) -> tuple[float, float, float, float]:
    """Fit delta_product = alpha + beta * delta_week via HuberRegressor.

    Returns (alpha, beta, residual_std, r2).
    """
    if not _HAS_SKLEARN:
        logger.warning("scikit-learn not available; falling back to OLS for translation regression")
        return _fit_ols_fallback(delta_week, delta_product)

    mask = np.isfinite(delta_week) & np.isfinite(delta_product)
    dw = delta_week[mask]
    dp = delta_product[mask]
    if len(dw) < 10:
        return 0.0, 1.0, float("nan"), 0.0

    X = dw.reshape(-1, 1)
    reg = HuberRegressor(epsilon=epsilon, max_iter=200)
    reg.fit(X, dp)
    alpha = float(reg.intercept_)
    beta = float(reg.coef_[0])

    resid = dp - (alpha + beta * dw)
    resid_std = float(np.std(resid))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((dp - np.mean(dp)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return alpha, beta, resid_std, r2


def _fit_ols_fallback(
    delta_week: np.ndarray,
    delta_product: np.ndarray,
) -> tuple[float, float, float, float]:
    mask = np.isfinite(delta_week) & np.isfinite(delta_product)
    dw = delta_week[mask]
    dp = delta_product[mask]
    if len(dw) < 10:
        return 0.0, 1.0, float("nan"), 0.0

    X = np.column_stack([np.ones_like(dw), dw])
    coef, res, _, _ = np.linalg.lstsq(X, dp, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = dp - (alpha + beta * dw)
    resid_std = float(np.std(resid))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((dp - np.mean(dp)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return alpha, beta, resid_std, r2


def translate_product(
    product_code: str,
    strip_type: str,
    market_forward_product: float,
    market_forward_week: float,
    model_fv_week: ProductFairValue,
    config: PostModelConfig,
    *,
    historical_delta_week: np.ndarray | None = None,
    historical_delta_product: np.ndarray | None = None,
    week_path_fvs: np.ndarray | None = None,
    historical_residuals: np.ndarray | None = None,
) -> TranslatedProduct:
    """Translate prompt month/quarter fair value from the prompt week model FV.

    Uses the change-regression method:
        translated_FV = market_forward + beta * (model_week_FV - market_forward_week)
    """
    tc = config.translation
    diag = TranslationDiagnostics()

    if historical_delta_week is not None and historical_delta_product is not None:
        n_obs = len(historical_delta_week)
        diag.n_observations = n_obs

        if n_obs < tc.min_history_days:
            if not tc.enable_fallback_shrinkage:
                raise ValueError(
                    f"Insufficient history ({n_obs} days) for translation of {product_code}. "
                    f"Need {tc.min_history_days}. Enable fallback_shrinkage in config to proceed."
                )
            alpha, beta, resid_std, r2 = _fit_translation_regression(
                historical_delta_week, historical_delta_product, tc.huber_epsilon,
            )
            beta *= tc.fallback_shrinkage_factor
            diag.method_used = "huber_regression_shrinkage_fallback"
            diag.translation_quality = "fallback"
        else:
            alpha, beta, resid_std, r2 = _fit_translation_regression(
                historical_delta_week, historical_delta_product, tc.huber_epsilon,
            )
            diag.method_used = "huber_regression"

            half = n_obs // 2
            if half >= 10:
                _, b1, _, _ = _fit_translation_regression(
                    historical_delta_week[:half], historical_delta_product[:half], tc.huber_epsilon,
                )
                _, b2, _, _ = _fit_translation_regression(
                    historical_delta_week[half:], historical_delta_product[half:], tc.huber_epsilon,
                )
                diag.beta_stability = 1.0 - abs(b1 - b2) / max(abs(beta), 1e-6)
                diag.sign_stability = 1.0 if np.sign(b1) == np.sign(b2) else 0.0
            else:
                diag.beta_stability = 0.5
                diag.sign_stability = 0.5

            if r2 > 0.3 and diag.beta_stability > tc.beta_stability_threshold:
                diag.translation_quality = "good"
            elif r2 > tc.r2_threshold:
                diag.translation_quality = "fair"
            else:
                diag.translation_quality = "weak"

        diag.alpha = alpha
        diag.beta = beta
        diag.residual_std = resid_std
        diag.rolling_r2 = r2
    else:
        beta = 1.0
        diag.beta = 1.0
        diag.method_used = "pass_through_no_history"
        diag.translation_quality = "no_history"
        resid_std = 0.0

    model_week_fv = model_fv_week.fair_value_mean
    translated_fv = market_forward_product + beta * (model_week_fv - market_forward_week)

    model_week_std = model_fv_week.fair_value_std or 0.0
    translated_std = float(np.sqrt((beta * model_week_std) ** 2 + resid_std ** 2))

    p10: float | None = None
    p50: float | None = None
    p90: float | None = None

    if week_path_fvs is not None and len(week_path_fvs) > 0:
        if historical_residuals is not None and len(historical_residuals) > 0:
            rng = np.random.default_rng(42)
            resid_draws = rng.choice(historical_residuals, size=len(week_path_fvs), replace=True)
        else:
            resid_draws = np.zeros(len(week_path_fvs))

        translated_paths = (
            market_forward_product
            + beta * (week_path_fvs - market_forward_week)
            + resid_draws
        )
        p10 = float(np.percentile(translated_paths, 10))
        p50 = float(np.percentile(translated_paths, 50))
        p90 = float(np.percentile(translated_paths, 90))
        translated_fv = float(np.mean(translated_paths))
        translated_std = float(np.std(translated_paths))
        diag.coverage_ratio = float(np.mean(
            (translated_paths >= (p10 or 0)) & (translated_paths <= (p90 or 0))
        ))

    edge = translated_fv - market_forward_product
    risk_premium = market_forward_product - translated_fv

    return TranslatedProduct(
        product_code=product_code,
        strip_type=strip_type,
        market_forward=market_forward_product,
        fair_value=translated_fv,
        std=translated_std,
        p10=p10,
        p50=p50,
        p90=p90,
        edge=edge,
        risk_premium_proxy=risk_premium,
        diagnostics=diag,
    )


def build_curve_translation(
    bundle: ForecastSnapshot,
    config: PostModelConfig,
) -> dict[str, TranslatedProduct]:
    """Build full curve translation for all products.

    Direct pathwise aggregation for prompt week (base/peak).
    Translation regression for month/quarter (requires market_forwards in bundle).
    """
    results: dict[str, TranslatedProduct] = {}

    if bundle.aggregation_level == "scenario_paths" and bundle.scenario_paths is not None:
        direct_fvs = compute_product_fair_values_pathwise(bundle, config)
    else:
        direct_fvs = compute_product_fair_values_point(bundle, config)

    week_base = direct_fvs.get("prompt_week_base")
    week_peak = direct_fvs.get("prompt_week_peak")

    if week_base is None and "next_day_base" in direct_fvs:
        week_base = direct_fvs["next_day_base"]
    if week_peak is None and "next_day_peak" in direct_fvs:
        week_peak = direct_fvs["next_day_peak"]

    week_path_fvs: np.ndarray | None = None
    if bundle.scenario_paths is not None and week_base is not None:
        paths_df = bundle.scenario_paths
        path_cols = [c for c in paths_df.columns if c.startswith("path_")]
        paths_matrix = paths_df[path_cols].values
        week_path_fvs = paths_matrix.mean(axis=0)

    for fv in direct_fvs.values():
        code = fv.product_code
        has_mkt = code in bundle.market_forwards
        mkt_fwd = bundle.market_forwards.get(code, fv.fair_value_mean)
        results[code] = TranslatedProduct(
            product_code=code,
            strip_type=fv.strip_type,
            market_forward=mkt_fwd,
            fair_value=fv.fair_value_mean,
            std=fv.fair_value_std or 0.0,
            p10=fv.fair_value_p10,
            p50=fv.fair_value_p50,
            p90=fv.fair_value_p90,
            edge=fv.fair_value_mean - mkt_fwd,
            risk_premium_proxy=mkt_fwd - fv.fair_value_mean,
            diagnostics=TranslationDiagnostics(
                method_used="direct_pathwise" if bundle.scenario_paths is not None else "point_mean",
                translation_quality="direct",
                coverage_ratio=1.0,
            ),
            has_market_forward=has_mkt,
        )

    for code in ["prompt_month_base", "prompt_month_peak", "prompt_quarter_base", "prompt_quarter_peak"]:
        strip = "peak" if "peak" in code else "base"
        mkt_fwd = bundle.market_forwards.get(code)
        if mkt_fwd is None:
            logger.info("No market forward for %s — skipping translation", code)
            continue

        ref_week = week_peak if strip == "peak" else week_base
        if ref_week is None:
            logger.warning("No week %s FV available for translating %s", strip, code)
            continue

        mkt_fwd_week = bundle.market_forwards.get(
            f"prompt_week_{strip}", ref_week.fair_value_mean
        )

        results[code] = translate_product(
            product_code=code,
            strip_type=strip,
            market_forward_product=mkt_fwd,
            market_forward_week=mkt_fwd_week,
            model_fv_week=ref_week,
            config=config,
            week_path_fvs=week_path_fvs,
        )

    return results
