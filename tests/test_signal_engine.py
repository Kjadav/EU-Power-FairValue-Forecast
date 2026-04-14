"""Tests for post-model signal engine."""

from __future__ import annotations

import numpy as np
import pytest

from trading_pipeline_utils.market.curve import TranslatedProduct, TranslationDiagnostics
from trading_pipeline_utils.market.signals import compute_all_signals, compute_signal
from trading_pipeline_utils.settings import PostModelConfig


def _make_product(
    code: str = "prompt_week_base",
    fv: float = 85.0,
    mkt: float = 80.0,
    std: float = 3.0,
    p10: float | None = 78.0,
    p90: float | None = 92.0,
    quality: str = "direct",
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
            method_used="direct_pathwise",
            translation_quality=quality,
            coverage_ratio=coverage,
        ),
    )


class TestComputeSignal:
    def test_long_direction_when_positive_edge(self):
        product = _make_product(fv=90.0, mkt=80.0, std=3.0)
        config = PostModelConfig()
        bt = {"overall_mae_eur_mwh": 10.0}
        sig = compute_signal(product, bt, config)
        assert sig.direction == "long"
        assert sig.edge > 0
        assert sig.edge_z > 0

    def test_short_direction_when_negative_edge(self):
        product = _make_product(fv=75.0, mkt=85.0, std=3.0)
        config = PostModelConfig()
        bt = {"overall_mae_eur_mwh": 10.0}
        sig = compute_signal(product, bt, config)
        assert sig.direction == "short"
        assert sig.edge < 0
        assert sig.edge_z < 0

    def test_flat_when_edge_small(self):
        product = _make_product(fv=80.1, mkt=80.0, std=5.0)
        config = PostModelConfig()
        bt = {"overall_mae_eur_mwh": 10.0}
        sig = compute_signal(product, bt, config)
        assert sig.direction == "flat"

    def test_confidence_bounded_zero_one(self):
        product = _make_product(fv=200.0, mkt=80.0, std=1.0)
        config = PostModelConfig()
        bt = {"overall_mae_eur_mwh": 5.0}
        sig = compute_signal(product, bt, config)
        assert 0.0 <= sig.confidence <= 1.0

    def test_signal_score_sign_matches_edge(self):
        product = _make_product(fv=90.0, mkt=80.0, std=3.0)
        config = PostModelConfig()
        bt = {"overall_mae_eur_mwh": 10.0}
        sig = compute_signal(product, bt, config)
        assert np.sign(sig.signal_score) == np.sign(sig.edge)

    def test_risk_premium_proxy_sign(self):
        product = _make_product(fv=90.0, mkt=80.0, std=3.0)
        config = PostModelConfig()
        bt = {"overall_mae_eur_mwh": 10.0}
        sig = compute_signal(product, bt, config)
        # edge = FV - mkt = +10, risk_premium = mkt - FV = -10
        assert sig.risk_premium_proxy == -sig.edge

    def test_weak_translation_reduces_confidence(self):
        good = _make_product(fv=90.0, mkt=80.0, std=3.0, quality="good")
        weak = _make_product(fv=90.0, mkt=80.0, std=3.0, quality="weak")
        config = PostModelConfig()
        bt = {"overall_mae_eur_mwh": 10.0}
        sig_good = compute_signal(good, bt, config)
        sig_weak = compute_signal(weak, bt, config)
        assert sig_good.confidence > sig_weak.confidence


class TestComputeAllSignals:
    def test_all_products_get_signals(self):
        products = {
            "prompt_week_base": _make_product("prompt_week_base"),
            "prompt_week_peak": _make_product("prompt_week_peak", fv=95.0, mkt=90.0),
        }
        config = PostModelConfig()
        bt = {"overall_mae_eur_mwh": 10.0}
        signals = compute_all_signals(products, bt, config)
        assert set(signals.keys()) == set(products.keys())
