from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from trading_pipeline_utils.types import AggregationLevel

FundamentalFeatureMode = Literal["residual_wind_solar_hydro", "load_wind_solar_hydro"]
MissingHoursPolicy = Literal["fail", "keep_na"]
SimulationMode = Literal["d_minus_1_recursive", "single_origin_week"]
WindowType = Literal["expanding", "rolling"]

def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its root mapping."""
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def default_config_path() -> Path:
    """Resolve the default config file relative to the repo root."""
    return Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml"

#buiilding dataclass for correct congig/typing in the different functions in pipeline

@dataclass
class DataVendorConfig:
    type: str
    base_url: str
    region: str
    resolution: str
    physical_flow_resolution: str
    include_physical_flows: bool
    on_physical_flow_error: str
    meta_filename: str
    bundle_single_series: dict[str, dict[str, str]]
    wind_forecast_sources: list[dict[str, str]]
    wind_generation_sources: list[dict[str, str]]
    solar_generation_actual_filter: str
    realized_generation_carrier_filters: list[str]
    generation_pumped_storage_filter: str
    table_value_columns: dict[str, str]
    physical_flow_value_column: str
    default_physical_flow_ids: list[str]
    api_key: str | None = None
    smard_min_history_hours: int = 8760
    smard_max_index_chunks: int = 120


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    min_confidence_enforced: float
    api_key: str | None = None


@dataclass
class PipelineSectionConfig:
    output_dir: str
    forecast_horizon_hours: int
    naive_method: str 
    signal_threshold_fraction: float


@dataclass
class PipelineConfig:
    data_vendor: DataVendorConfig
    llm: LLMConfig
    pipeline: PipelineSectionConfig
    logging_level: str = "INFO"

@dataclass
class PeakDefinition:
    weekdays_only: bool = True
    hour_start: int = 8
    hour_end: int = 20


@dataclass
class ProductConfig:
    """Prompt product definitions for curve translation."""

    peak: PeakDefinition = field(default_factory=PeakDefinition)
    delivery_timezone: str = "Europe/Berlin"
    prompt_week_hours: int = 168
    prompt_month_hours: int = 720
    prompt_quarter_hours: int = 2160


@dataclass
class TranslationConfig:
    rolling_window_days: int = 180
    min_history_days: int = 60
    huber_epsilon: float = 1.35
    beta_stability_threshold: float = 0.5
    r2_threshold: float = 0.1
    enable_fallback_shrinkage: bool = False
    fallback_shrinkage_factor: float = 0.5


@dataclass
class SignalConfig:
    z_cap: float = 3.0
    entry_threshold: float = 0.75
    risk_budget: float = 1.0
    max_units: float = 10.0
    epsilon: float = 1e-6


@dataclass
class InvalidationConfig:
    sigma_shift_threshold: float = 2.0
    beta_stability_min: float = 0.3
    r2_min: float = 0.05
    max_uncertainty_multiple: float = 5.0
    max_signal_age_hours: int = 48
    min_coverage_ratio: float = 0.5
    skill_degradation_threshold: float = 0.5


@dataclass
class LLMInsightConfig:
    enabled: bool = True
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    max_tokens: int = 8192
    timeout_seconds: float = 90.0
    api_key: str | None = None


@dataclass
class PostModelConfig:
    products: ProductConfig = field(default_factory=ProductConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    invalidation: InvalidationConfig = field(default_factory=InvalidationConfig)
    llm_insight: LLMInsightConfig = field(default_factory=LLMInsightConfig)
    aggregation_level: AggregationLevel = "scenario_paths"
    market_name: str = "DE-LU"
    enable_quantile_approximation: bool = False

@dataclass(frozen=True)
class ColumnMap:
    """Canonical column names after bundle load and validation."""

    timestamp: str = "timestamp"
    price_da: str = "price_da"
    residual_load_fcst: str = "residual_load_fcst"
    residual_load_act: str = "residual_load_act"
    wind_fcst: str = "wind_fcst"
    wind_act: str = "wind_act"
    solar_fcst: str = "solar_fcst"
    solar_act: str = "solar_act"
    hydro_fcst: str = "hydro_fcst"
    hydro_act: str = "hydro_act"
    load_fcst: str = "load_fcst"
    load_act: str = "load_act"


@dataclass
class ValidationConfig:
    delivery_timezone: str = "Europe/Berlin"
    missing_hours_policy: MissingHoursPolicy = "fail"
    require_strictly_hourly: bool = True
    drop_exact_row_duplicates: bool = True


@dataclass
class FeatureConfig:
    """Toggle residual vs load fundamentals (avoid perfect collinearity in v1)."""

    fundamental_feature_mode: FundamentalFeatureMode = "load_wind_solar_hydro"
    include_optional_holiday_flag: bool = False
    holiday_country: str = "DE"
    include_forecast_ratios: bool = True
    ratio_eps: float = 1e-6


@dataclass
class TargetTransformConfig:
    scale_floor: float = 10.0


@dataclass
class LightGBMFixedParams:
    boosting_type: str = "gbdt"
    learning_rate: float = 0.03
    n_estimators: int = 3000
    num_leaves: int = 63
    max_depth: int = 8
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l1: float = 0.5
    lambda_l2: float = 5.0
    min_gain_to_split: float = 0.05
    max_bin: int = 255
    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int = 200
    verbose: int = -1


@dataclass
class HyperparameterSearchConfig:
    enabled: bool = True
    max_trials: int = 48
    random_state: int = 42
    num_leaves_choices: tuple[int, ...] = (31, 63, 127)
    max_depth_choices: tuple[int, ...] = (6, 8, 10)
    min_data_in_leaf_choices: tuple[int, ...] = (100, 200, 500)
    feature_fraction_choices: tuple[float, ...] = (0.7, 0.85, 1.0)
    bagging_fraction_choices: tuple[float, ...] = (0.7, 0.85, 1.0)
    lambda_l1_choices: tuple[float, ...] = (0.0, 0.5, 2.0)
    lambda_l2_choices: tuple[float, ...] = (1.0, 5.0, 10.0)
    min_gain_to_split_choices: tuple[float, ...] = (0.0, 0.05, 0.1)


@dataclass
class SampleWeightConfig:
    neg_price_extra: float = 1.5
    half_life_days: float = 365.0


@dataclass
class BacktestConfig:
    validation_block_hours: int = 28 * 24
    gap_hours: int = 24
    origin_step_hours: int = 24
    window_type: WindowType = "expanding"
    rolling_train_hours: int | None = 24 * 365 * 3
    holdout_tail_hours: int | None = 28 * 24


@dataclass
class SimulationConfig:
    n_paths: int = 1000
    horizon_hours: int = 7 * 24
    sigma_floor: float = 1e-3
    deterministic: bool = False
    use_gaussian_shocks: bool = False
    perfect_foresight_mode: bool = False
    simulation_mode: SimulationMode = "d_minus_1_recursive"
    peak_local_weekdays_only: bool = True
    peak_local_hour_start: int = 8
    peak_local_hour_end: int = 20
    reference_market_price: float | None = None
    z_clip: float = 3.5
    price_floor_eur: float = -500.0
    price_ceil_eur: float = 1000.0


@dataclass
class ExplainabilityConfig:
    shap_max_samples: int = 500
    plot_top_n: int = 20


@dataclass
class OutputConfig:
    base_dir: Path = field(default_factory=lambda: Path("data/processed/da_model"))
    fold_metrics_csv: str = "fold_metrics.csv"
    holdout_metrics_json: str = "holdout_metrics.json"
    oof_predictions_parquet: str = "oof_predictions.parquet"
    final_forecasts_parquet: str = "final_forecasts.parquet"
    simulated_paths_parquet: str = "simulated_paths.parquet"
    weekly_distributions_csv: str = "weekly_distributions.csv"
    models_joblib: str = "fitted_models.joblib"
    config_snapshot_json: str = "config_snapshot.json"
    shap_summary_csv: str = "shap_summary.csv"
    calibration_plot: str = "calibration.png"
    backtest_chart: str = "backtest_mae.png"
    feature_importance_csv: str = "feature_importance.csv"


@dataclass
class ModelPipelineConfig:
    columns: ColumnMap = field(default_factory=ColumnMap)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    target: TargetTransformConfig = field(default_factory=TargetTransformConfig)
    lgbm: LightGBMFixedParams = field(default_factory=LightGBMFixedParams)
    hyperparam_search: HyperparameterSearchConfig = field(default_factory=HyperparameterSearchConfig)
    sample_weight: SampleWeightConfig = field(default_factory=SampleWeightConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    run_xgboost_benchmark: bool = False
    separate_model_per_hour: bool = False
    log_level: str = "INFO"
    clip_price_warnings_threshold_eur: float | None = None

def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"config missing required key: {key!r}")
    return d[key]


def _dv_from_dict(d: dict[str, Any]) -> DataVendorConfig:
    return DataVendorConfig(
        type=str(_require(d, "type")),
        base_url=str(_require(d, "base_url")).rstrip("/"),
        region=str(_require(d, "region")),
        resolution=str(_require(d, "resolution")),
        physical_flow_resolution=str(d.get("physical_flow_resolution", "quarterhour")),
        include_physical_flows=bool(d.get("include_physical_flows", False)),
        on_physical_flow_error=str(d.get("on_physical_flow_error", "skip")),
        meta_filename=str(d.get("meta_filename", "_smard_bundle_meta.json")),
        bundle_single_series=dict(d.get("bundle_single_series") or {}),
        wind_forecast_sources=list(d.get("wind_forecast_sources") or []),
        wind_generation_sources=list(d.get("wind_generation_sources") or []),
        solar_generation_actual_filter=str(_require(d, "solar_generation_actual_filter")),
        realized_generation_carrier_filters=list(_require(d, "realized_generation_carrier_filters")),
        generation_pumped_storage_filter=str(_require(d, "generation_pumped_storage_filter")),
        table_value_columns=dict(_require(d, "table_value_columns")),
        physical_flow_value_column=str(d.get("physical_flow_value_column", "physical_flow_mw")),
        default_physical_flow_ids=list(d.get("default_physical_flow_ids") or []),
        api_key=d.get("api_key"),
        smard_min_history_hours=int(d.get("smard_min_history_hours", 8760)),
        smard_max_index_chunks=int(d.get("smard_max_index_chunks", 120)),
    )


def _llm_from_dict(d: dict[str, Any]) -> LLMConfig:
    return LLMConfig(
        provider=str(d.get("provider", "gemini")),
        model=str(_require(d, "model")),
        temperature=float(d.get("temperature", 0.2)),
        max_tokens=int(d.get("max_tokens", 1024)),
        timeout_seconds=float(d.get("timeout_seconds", 60.0)),
        min_confidence_enforced=float(d.get("min_confidence_enforced", 0.85)),
        api_key=d.get("api_key"),
    )


def _pipe_from_dict(d: dict[str, Any]) -> PipelineSectionConfig:
    return PipelineSectionConfig(
        output_dir=str(_require(d, "output_dir")),
        forecast_horizon_hours=int(d.get("forecast_horizon_hours", 24)),
        naive_method=str(d.get("naive_method", "seasonal_24h")),
        signal_threshold_fraction=float(d.get("signal_threshold_fraction", 0.02)),
    )


def parse_pipeline_config(raw: dict[str, Any]) -> PipelineConfig:
    """Parse a raw config dict into a fully typed ``PipelineConfig``."""
    return PipelineConfig(
        data_vendor=_dv_from_dict(dict(_require(raw, "data_vendor"))),
        llm=_llm_from_dict(dict(_require(raw, "llm"))),
        pipeline=_pipe_from_dict(dict(_require(raw, "pipeline"))),
        logging_level=str(raw.get("logging_level", "INFO")),
    )

#env mgmt
def _merge_env_secrets(raw: dict[str, Any]) -> None:
    """Inject API keys from environment variables when not set in YAML."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    vendor_key = os.environ.get("DATA_VENDOR_API_KEY")

    if gemini_key:
        raw.setdefault("llm", {}).setdefault("api_key", gemini_key)
    if vendor_key:
        raw.setdefault("data_vendor", {}).setdefault("api_key", vendor_key)


def load_config(config_path: Path | None = None) -> PipelineConfig:
    """Load YAML, merge env secrets, and return a typed ``PipelineConfig``."""
    path = config_path or default_config_path()
    raw = load_yaml(path)
    _merge_env_secrets(raw)
    return parse_pipeline_config(raw)
