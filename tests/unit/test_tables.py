"""Tests for deterministic desk-action tables and rule-based summaries."""

from __future__ import annotations

import pytest

from trading_pipeline_utils.reporting.tables import build_desk_action_table, build_deterministic_summaries
from trading_pipeline_utils.market.curve import TranslatedProduct, TranslationDiagnostics
from trading_pipeline_utils.market.signals import Signal
from trading_pipeline_utils.market.invalidation import InvalidationResult


def _make_product(code="test_prod") -> TranslatedProduct:
    return TranslatedProduct(
        product_code=code, strip_type="base",
        market_forward=80.0, fair_value=85.0, std=5.0,
        p10=75.0, p50=85.0, p90=95.0,
        edge=5.0, risk_premium_proxy=-5.0,
        diagnostics=TranslationDiagnostics(translation_quality="direct", coverage_ratio=0.8),
    )


def _make_signal(code="test_prod") -> Signal:
    return Signal(
        product_code=code, edge=5.0, edge_z=1.0,
        risk_premium_proxy=-5.0, confidence=0.7,
        signal_score=0.7, direction="long",
        suggested_position_units=1.0,
        suggested_expression="Long test_prod",
        model_skill_weight=0.8, translation_stability_weight=1.0,
        coverage_weight=0.8, distribution_weight=0.9,
    )


def test_desk_action_table():
    prods = {"test_prod": _make_product()}
    sigs = {"test_prod": _make_signal()}
    invs = {"test_prod": InvalidationResult(product_code="test_prod", invalidation_flag=False)}
    table = build_desk_action_table(prods, sigs, invs)
    assert len(table) == 1
    assert table.iloc[0]["suggested_direction"] == "long"


def test_deterministic_summaries():
    prods = {"test_prod": _make_product()}
    sigs = {"test_prod": _make_signal()}
    invs = {"test_prod": InvalidationResult(product_code="test_prod", invalidation_flag=False)}
    summaries = build_deterministic_summaries(sigs, prods, invs)
    assert len(summaries) == 1
    assert "Long" in summaries[0]
