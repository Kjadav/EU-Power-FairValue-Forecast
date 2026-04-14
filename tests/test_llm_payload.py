"""Tests for LLM payload construction and deterministic fallback."""

from __future__ import annotations

import json

import pytest

from trading_pipeline_utils.llm.schemas import LLMInsightResult
from trading_pipeline_utils.llm.summarizer import (
    _deterministic_fallback,
    build_signal_payload,
    generate_llm_insights,
)
from trading_pipeline_utils.market.curve import TranslatedProduct, TranslationDiagnostics
from trading_pipeline_utils.market.invalidation import InvalidationResult
from trading_pipeline_utils.market.shapes import ShapeView
from trading_pipeline_utils.market.signals import Signal
from trading_pipeline_utils.settings import PostModelConfig


def _make_signal(code: str, edge: float = 5.0, direction: str = "long") -> Signal:
    return Signal(
        product_code=code,
        edge=edge,
        edge_z=edge / 3.0,
        risk_premium_proxy=-edge,
        confidence=0.6,
        signal_score=0.6 if edge > 0 else -0.6,
        direction=direction,
        suggested_position_units=1.0,
        suggested_expression="test",
        model_skill_weight=0.7,
        translation_stability_weight=0.9,
        coverage_weight=0.8,
        distribution_weight=0.7,
    )


def _make_product(code: str, fv: float = 85.0, mkt: float = 80.0) -> TranslatedProduct:
    return TranslatedProduct(
        product_code=code,
        strip_type="base",
        market_forward=mkt,
        fair_value=fv,
        std=3.0,
        p10=75.0, p50=fv, p90=95.0,
        edge=fv - mkt,
        risk_premium_proxy=mkt - fv,
        diagnostics=TranslationDiagnostics(
            method_used="direct_pathwise",
            translation_quality="direct",
            coverage_ratio=0.8,
        ),
    )


def _sample_payload():
    products = {"prompt_week_base": _make_product("prompt_week_base")}
    signals = {"prompt_week_base": _make_signal("prompt_week_base")}
    shapes = [
        ShapeView(
            "peak_vs_base", "prompt_week_peak", "prompt_week_base",
            13.0, 13.0, 0.0, 0.0, "flat", "Flat",
        ),
    ]
    invalidations = {
        "prompt_week_base": InvalidationResult(product_code="prompt_week_base", invalidation_flag=False),
    }
    config = PostModelConfig()
    return build_signal_payload(
        products, signals, shapes, invalidations,
        {"overall_mae_eur_mwh": 10.0, "q10_q90_coverage": 0.78},
        config, "2026-04-13",
    )


class TestPayloadSchema:
    def test_payload_has_required_fields(self):
        payload = _sample_payload()
        assert "as_of_date" in payload
        assert "market_name" in payload
        assert "outright_signals" in payload
        assert "shape_signals" in payload
        assert "backtest_summary" in payload

    def test_outright_signal_fields(self):
        payload = _sample_payload()
        sig = payload["outright_signals"][0]
        required_fields = {
            "product", "direction", "edge", "edge_z", "confidence",
            "signal_score", "fair_value", "market_forward", "std",
            "risk_premium_proxy", "translation_method", "translation_quality",
            "invalidation_flag",
        }
        assert required_fields.issubset(set(sig.keys()))

    def test_payload_serializable(self):
        payload = _sample_payload()
        serialized = json.dumps(payload, default=str)
        assert len(serialized) > 0
        reparsed = json.loads(serialized)
        assert reparsed["as_of_date"] == "2026-04-13"


class TestDeterministicFallback:
    def test_fallback_returns_all_fields(self):
        payload = _sample_payload()
        result = _deterministic_fallback(payload)
        assert isinstance(result, LLMInsightResult)
        assert result.source == "deterministic_fallback"
        assert len(result.executive_summary) > 0
        assert isinstance(result.outright_views, list)
        assert isinstance(result.key_risks, list)

    def test_fallback_when_no_api_key(self):
        payload = _sample_payload()
        config = PostModelConfig()
        config.llm_insight.api_key = None
        result = generate_llm_insights(payload, config)
        assert result.source == "deterministic_fallback"

    def test_fallback_when_disabled(self):
        payload = _sample_payload()
        config = PostModelConfig()
        config.llm_insight.enabled = False
        result = generate_llm_insights(payload, config)
        assert result.source == "deterministic_fallback"

    def test_fallback_with_no_signals(self):
        payload = {
            "as_of_date": "2026-04-13",
            "market_name": "DE-LU",
            "outright_signals": [],
            "shape_signals": [],
            "backtest_summary": {},
        }
        result = _deterministic_fallback(payload)
        assert "no strong outright signals" in result.executive_summary.lower()
