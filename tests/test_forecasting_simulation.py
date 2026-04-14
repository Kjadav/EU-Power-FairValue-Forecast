"""Simulation mode guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_pipeline_utils.features.transforms import ArcsinhPriceTransform
from trading_pipeline_utils.forecasting.simulation import simulate_paths_next_week
from trading_pipeline_utils.settings import ModelPipelineConfig


def test_single_origin_without_perfect_foresight_raises() -> None:
    pytest.importorskip("lightgbm")
    cfg = ModelPipelineConfig()
    cfg.simulation.simulation_mode = "single_origin_week"
    cfg.simulation.perfect_foresight_mode = False
    cfg.simulation.n_paths = 2
    tfm = ArcsinhPriceTransform(10.0)
    tfm.fit(np.array([50.0, 60.0]))
    H = 24
    idx = pd.date_range("2025-01-01", periods=H, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="single_origin_week"):
        simulate_paths_next_week(
            np.zeros(H),
            np.full(H, -1.0),
            np.full(H, 1.0),
            tfm,
            np.array([0.0, 0.1]),
            np.zeros(H, dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
            cfg,
            future_index=idx,
        )
