from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from trading_pipeline_utils.settings import ModelPipelineConfig

try:
    import holidays as _holidays_lib

    _HAS_HOLIDAYS = True
except ImportError:
    _holidays_lib = None
    _HAS_HOLIDAYS = False

logger = logging.getLogger(__name__)

# Longest lookback  implied by column names; used elsewhere for leakage checks.
FEATURE_GROUPS_MAX_LAG_HOURS: dict[str, int] = {
    "price_lag_168": 168,
    "price_roll_168": 168,
    "rv_168": 169,
    "fe_roll_168": 168,
}


def _safe_div(num: pd.Series, den: pd.Series, eps: float) -> pd.Series:
    """Elementwise num / (den + eps); avoids divide-by-zero on small loads."""
    return num / (den + eps)


def compute_residual_load(df: pd.DataFrame, c: object) -> pd.Series:
    """Forecast load minus wind, solar, and hydro forecasts (single bidding-zone style)."""
    return df[c.load_fcst] - df[c.wind_fcst] - df[c.solar_fcst] - df[c.hydro_fcst]


def compute_residual_load_act(df: pd.DataFrame, c: object) -> pd.Series:
    """Actual load minus actual wind/solar/hydro; NaN row if any actual column is missing."""
    cols = [c.load_act, c.wind_act, c.solar_act, c.hydro_act]
    if not all(x in df.columns for x in cols):
        return pd.Series(np.nan, index=df.index)
    return df[c.load_act] - df[c.wind_act] - df[c.solar_act] - df[c.hydro_act]


def _evening_hourly_spread(evening_prices: pd.Series) -> float:
    """One number per local day: max − min DA price across that day’s evening hours."""
    if len(evening_prices) <= 1:
        return float("nan")
    return float(evening_prices.max() - evening_prices.min())


def _map_previous_local_day(
    daily_by_local_date: pd.Series,
    local_dates: pd.Series,
) -> pd.Series:
    return local_dates.map(daily_by_local_date.shift(1))


def _ordered_category(series: pd.Series, categories: list, index: pd.Index) -> pd.Series:
    """Wrap *series* as a fixed-cardinality pandas ``category`` on *index*."""
    return pd.Series(pd.Categorical(series, categories=categories), index=index).astype("category")


def build_feature_matrix(
    df: pd.DataFrame,
    config: ModelPipelineConfig,
    *,
    train_abs_price_p95: float | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Stack model inputs aligned to *df*’s index.

    Parameters
    ----------
    df
        Hourly frame with DA price, fundamentals, and precomputed local calendar columns.
    config
        Column names, feature toggles, and simulation/validation settings.
    train_abs_price_p95
        If set, ``spike_count_168`` counts hours in the last week where |price| exceeded
        this threshold (typically the training-set 95th percentile of |price|).

    Returns
    -------
    features
        All engineered columns.
    numeric_cols / cat_cols
        Column names split by dtype for the trainer (e.g. LightGBM categorical handling).
    """
    c = config.columns
    fc = config.features
    price = df[c.price_da].astype(np.float64)
    out = pd.DataFrame(index=df.index)

    # --- Price returns and volatility-style scalars (all shifted so t uses < t) ---
    y_for_vol = price
    ret1 = y_for_vol.diff(1)
    out["rv_24"] = np.sqrt((ret1**2).rolling(24, min_periods=24).mean()).shift(1)
    out["rv_168"] = np.sqrt((ret1**2).rolling(168, min_periods=168).mean()).shift(1)
    out["abs_ret_24"] = ret1.abs().rolling(24, min_periods=24).mean().shift(1)
    out["price_range_24"] = (
        y_for_vol.rolling(24, min_periods=24).max().shift(1)
        - y_for_vol.rolling(24, min_periods=24).min().shift(1)
    )
    out["price_range_168"] = (
        y_for_vol.rolling(168, min_periods=168).max().shift(1)
        - y_for_vol.rolling(168, min_periods=168).min().shift(1)
    )
    if train_abs_price_p95 is not None:
        thr = float(train_abs_price_p95)
        out["spike_count_168"] = (
            (y_for_vol.abs() > thr).astype(np.float64).rolling(168, min_periods=168).sum().shift(1)
        )
    else:
        out["spike_count_168"] = np.nan
    out["neg_price_count_168"] = (
        (y_for_vol < 0).astype(np.float64).rolling(168, min_periods=168).sum().shift(1)
    )
    out["max_abs_ret_24"] = ret1.abs().rolling(24, min_periods=24).max().shift(1)

    # --- Lags and rolling moments of DA price ---
    for lag in (1, 2, 3, 6, 12, 24, 48, 72, 168):
        out[f"price_lag_{lag}"] = price.shift(lag)
    for w in (24, 48, 168):
        out[f"price_roll_mean_{w}"] = price.rolling(w, min_periods=w).mean().shift(1)
        out[f"price_roll_std_{w}"] = price.rolling(w, min_periods=w).std().shift(1)
    for w in (24, 168):
        out[f"price_roll_min_{w}"] = price.rolling(w, min_periods=w).min().shift(1)
        out[f"price_roll_max_{w}"] = price.rolling(w, min_periods=w).max().shift(1)

    # --- Previous *local delivery day* stats (calendar is delivery TZ, not UTC) ---
    tz = config.validation.delivery_timezone
    loc = df.index.tz_convert(tz)
    local_date = pd.Series(loc.date, index=df.index, dtype="object")
    tmp = pd.DataFrame({"_p": price, "_ld": local_date}, index=df.index)
    loc_h = loc.hour
    loc_dow = loc.dayofweek
    is_wd = loc_dow < 5
    peak_m = (
        is_wd
        & (loc_h >= config.simulation.peak_local_hour_start)
        & (loc_h < config.simulation.peak_local_hour_end)
    )
    evening_m = is_wd & (loc_h >= 18) & (loc_h <= 21)

    daily_b = tmp.groupby("_ld", sort=True)["_p"].mean().sort_index()
    daily_peak = tmp.loc[peak_m].groupby("_ld", sort=True)["_p"].mean().sort_index()
    daily_min = tmp.groupby("_ld", sort=True)["_p"].min().sort_index()
    daily_max = tmp.groupby("_ld", sort=True)["_p"].max().sort_index()
    daily_eve = (
        tmp.loc[evening_m].groupby("_ld", sort=True)["_p"].apply(_evening_hourly_spread).sort_index()
    )

    out["prev_day_baseload"] = _map_previous_local_day(daily_b, local_date)
    out["prev_day_peakload"] = _map_previous_local_day(daily_peak, local_date)
    out["prev_day_min"] = _map_previous_local_day(daily_min, local_date)
    out["prev_day_max"] = _map_previous_local_day(daily_max, local_date)
    out["prev_day_evening_ramp"] = _map_previous_local_day(daily_eve, local_date)

    # --- Calendar position as ordered categories (for tree models) ---
    idx = df.index
    out["delivery_hour_local_cat"] = _ordered_category(df["delivery_hour_local"], list(range(24)), idx)
    out["delivery_dow_local_cat"] = _ordered_category(
        df["delivery_day_of_week_local"], list(range(7)), idx
    )
    out["delivery_month_local_cat"] = _ordered_category(
        df["delivery_month_local"], list(range(1, 13)), idx
    )
    out["is_weekend_local_cat"] = _ordered_category(df["is_weekend_local"], [0, 1], idx)
    out["hour_of_week_local_cat"] = _ordered_category(
        df["hour_of_week_local"], list(range(24 * 7)), idx
    )

    if fc.include_optional_holiday_flag:
        if _HAS_HOLIDAYS:
            cal = _holidays_lib.country_holidays(fc.holiday_country)
            dloc = df.index.tz_convert(tz).date
            out["is_public_holiday_local_cat"] = _ordered_category(
                pd.Series([1 if d in cal else 0 for d in dloc], index=idx),
                [0, 1],
                idx,
            )
        else:
            logger.warning(
                "holidays package not installed; is_public_holiday_local_cat set to 0",
            )
            out["is_public_holiday_local_cat"] = _ordered_category(
                pd.Series(np.zeros(len(df), dtype=np.int8), index=idx),
                [0, 1],
                idx,
            )

    # --- Fundamentals: either raw load + renewables or residual load + renewables ---
    if fc.fundamental_feature_mode == "residual_wind_solar_hydro":
        if c.residual_load_fcst not in df.columns:
            out["residual_load_fcst"] = compute_residual_load(df, c)
        else:
            out["residual_load_fcst"] = df[c.residual_load_fcst]
        for name in (c.wind_fcst, c.solar_fcst, c.hydro_fcst):
            out[name] = df[name]
    else:
        for name in (c.load_fcst, c.wind_fcst, c.solar_fcst, c.hydro_fcst):
            out[name] = df[name]

    out["renewable_sum_fcst"] = df[c.wind_fcst] + df[c.solar_fcst] + df[c.hydro_fcst]

    if fc.fundamental_feature_mode == "residual_wind_solar_hydro":
        r_net = compute_residual_load(df, c)
        out["residual_load_fcst_ramp_1h"] = r_net.diff(1).shift(1)
        out["residual_load_fcst_chg_24h"] = (r_net - r_net.shift(24)).shift(1)

    if fc.fundamental_feature_mode == "load_wind_solar_hydro":
        fcst_cols = [x for x in (c.load_fcst, c.wind_fcst, c.solar_fcst, c.hydro_fcst) if x in df.columns]
    else:
        fcst_cols = [x for x in (c.wind_fcst, c.solar_fcst, c.hydro_fcst) if x in df.columns]
    for col in fcst_cols:
        s = df[col].astype(np.float64)
        out[f"{col}_ramp_1h"] = s.diff(1).shift(1)
        out[f"{col}_chg_24h"] = (s - s.shift(24)).shift(1)

    if fc.include_forecast_ratios:
        den = df[c.load_fcst].astype(np.float64)
        eps = fc.ratio_eps
        out["wind_share_of_load"] = _safe_div(df[c.wind_fcst].astype(np.float64), den, eps)
        out["solar_share_of_load"] = _safe_div(df[c.solar_fcst].astype(np.float64), den, eps)
        out["hydro_share_of_load"] = _safe_div(df[c.hydro_fcst].astype(np.float64), den, eps)

    # --- Realised generation: lags and forecast-error features when actuals exist ---
    act_map: dict[str, str] = {
        c.wind_act: "wind",
        c.solar_act: "solar",
        c.hydro_act: "hydro",
        c.load_act: "load",
    }
    fcst_for_short = {
        "wind": c.wind_fcst,
        "solar": c.solar_fcst,
        "hydro": c.hydro_fcst,
        "load": c.load_fcst,
    }
    for raw_col, short in act_map.items():
        if raw_col not in df.columns:
            continue
        a = df[raw_col].astype(np.float64)
        for lag in (1, 24, 168):
            out[f"{short}_act_lag_{lag}"] = a.shift(lag)
        fc_col = fcst_for_short[short]
        fe = a - df[fc_col].astype(np.float64)
        for lag in (1, 24, 168):
            out[f"{short}_fe_lag_{lag}"] = fe.shift(lag)
        out[f"{short}_fe_roll_mean_24"] = fe.rolling(24, min_periods=24).mean().shift(1)
        out[f"{short}_fe_roll_std_24"] = fe.rolling(24, min_periods=24).std().shift(1)
        out[f"{short}_fe_roll_mean_168"] = fe.rolling(168, min_periods=168).mean().shift(1)
        out[f"{short}_fe_roll_std_168"] = fe.rolling(168, min_periods=168).std().shift(1)

    numeric_cols = [x for x in out.columns if not str(out[x].dtype).startswith("category")]
    cat_cols = [x for x in out.columns if str(out[x].dtype).startswith("category")]
    return out, numeric_cols, cat_cols
