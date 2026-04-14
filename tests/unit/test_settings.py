"""Tests for pipeline configuration loading and validation."""

from __future__ import annotations

import pytest

from trading_pipeline_utils.settings import (
    PipelineConfig,
    PostModelConfig,
    ModelPipelineConfig,
    load_yaml,
    parse_pipeline_config,
    default_config_path,
    PeakDefinition,
    SignalConfig,
)


def test_default_config_path_exists():
    path = default_config_path()
    assert path.exists(), f"Default config not found: {path}"


def test_load_yaml():
    raw = load_yaml(default_config_path())
    assert isinstance(raw, dict)
    assert "data_vendor" in raw
    assert "llm" in raw


def test_parse_pipeline_config():
    raw = load_yaml(default_config_path())
    config = parse_pipeline_config(raw)
    assert isinstance(config, PipelineConfig)
    assert config.data_vendor.type == "smard"
    assert config.llm.model == "gemini-2.5-flash"


def test_postmodel_config_defaults():
    cfg = PostModelConfig()
    assert cfg.market_name == "DE-LU"
    assert cfg.signals.z_cap == 3.0
    assert cfg.invalidation.beta_stability_min == 0.3
    assert cfg.llm_insight.provider == "gemini"


def test_model_pipeline_config_defaults():
    cfg = ModelPipelineConfig()
    assert cfg.columns.price_da == "price_da"
    assert cfg.lgbm.n_estimators == 3000
    assert cfg.simulation.n_paths == 1000


def test_peak_definition_defaults():
    peak = PeakDefinition()
    assert peak.weekdays_only is True
    assert peak.hour_start == 8
    assert peak.hour_end == 20
