"""Tests for post-model invalidation logic."""

from __future__ import annotations

import pytest

from trading_pipeline_utils.settings import PostModelConfig
from trading_pipeline_utils.market.curve import TranslatedProduct, TranslationDiagnostics
from trading_pipeline_utils.market.invalidation import evaluate_invalidation, evaluate_all_invalidations
from trading_pipeline_utils.market.signals import Signal


def _make_signal(
    code: str = "prompt_week_base",
    edge: float = 5.0,
    edge_z: float = 1.5,
    confidence: float = 0.7,
    skill_weight: float = 0.75,
) -> Signal:
    return Signal(
        product_code=code,
        edge=edge,
        edge_z=edge_z,
        risk_premium_proxy=-edge,
        confidence=confidence,
        signal_score=confidence if edge > 0 else -confidence,
        direction="long" if edge > 0 else "short",
        suggested_position_units=1.0,
        suggested_expression="test",
        model_skill_weight=skill_weight,
        translation_stability_weight=0.9,
        coverage_weight=0.8,
        distribution_weight=0.7,
    )


def _make_product(
    code: str = "prompt_week_base",
    fv: float = 85.0,
    mkt: float = 80.0,
    std: float = 3.0,
    p10: float | None = 75.0,
    p90: float | None = 95.0,
    beta_stability: float = 0.8,
    r2: float = 0.5,
    quality: str = "good",
    coverage: float = 0.8,
) -> TranslatedProduct:
    return TranslatedProduct(
        product_code=code,
        strip_type="base",
        market_forward=mkt,
        fair_value=fv,
        std=std,
        p10=p10,
        p50=fv,
        p90=p90,
        edge=fv - mkt,
        risk_premium_proxy=mkt - fv,
        diagnostics=TranslationDiagnostics(
            beta_stability=beta_stability,
            rolling_r2=r2,
            translation_quality=quality,
            coverage_ratio=coverage,
        ),
    )


class TestInvalidation:
    def test_no_invalidation_when_healthy(self):
        sig = _make_signal()
        prod = _make_product()
        config = PostModelConfig()
        result = evaluate_invalidation(sig, prod, config)
        assert not result.invalidation_flag
        assert result.hold_or_exit == "hold"

    def test_fair_value_shift_triggers(self):
        sig = _make_signal()
        prod = _make_product(fv=85.0, std=3.0)
        config = PostModelConfig()
        result = evaluate_invalidation(sig, prod, config, prior_fair_value=70.0)
        assert result.invalidation_flag
        assert any("shifted" in r for r in result.invalidation_reasons)

    def test_beta_stability_triggers(self):
        sig = _make_signal()
        prod = _make_product(beta_stability=0.1, quality="fair")
        config = PostModelConfig()
        result = evaluate_invalidation(sig, prod, config)
        assert result.invalidation_flag
        assert any("Beta stability" in r for r in result.invalidation_reasons)

    def test_r2_triggers(self):
        sig = _make_signal()
        prod = _make_product(r2=0.01, quality="weak")
        config = PostModelConfig()
        result = evaluate_invalidation(sig, prod, config)
        assert result.invalidation_flag
        assert any("R²" in r for r in result.invalidation_reasons)

    def test_signal_age_triggers(self):
        sig = _make_signal()
        prod = _make_product()
        config = PostModelConfig()
        result = evaluate_invalidation(sig, prod, config, signal_age_hours=100.0)
        assert result.invalidation_flag
        assert any("age" in r.lower() for r in result.invalidation_reasons)

    def test_model_skill_triggers(self):
        sig = _make_signal(skill_weight=0.2)
        prod = _make_product()
        config = PostModelConfig()
        result = evaluate_invalidation(sig, prod, config)
        assert result.invalidation_flag
        assert any("skill" in r.lower() for r in result.invalidation_reasons)

    def test_multiple_triggers_escalate_to_exit(self):
        sig = _make_signal(skill_weight=0.2, edge_z=0.1, confidence=0.1)
        prod = _make_product(beta_stability=0.1, r2=0.01, quality="weak", coverage=0.1)
        config = PostModelConfig()
        result = evaluate_invalidation(sig, prod, config, prior_fair_value=50.0)
        assert result.invalidation_flag
        assert result.hold_or_exit == "exit"
        assert len(result.invalidation_reasons) >= 3

    def test_direct_product_skips_translation_checks(self):
        sig = _make_signal()
        prod = _make_product(beta_stability=0.0, r2=0.0, quality="direct")
        config = PostModelConfig()
        result = evaluate_invalidation(sig, prod, config)
        assert not any("Beta" in r for r in result.invalidation_reasons)
        assert not any("R²" in r for r in result.invalidation_reasons)
