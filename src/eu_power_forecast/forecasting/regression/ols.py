"""Ordinary least squares via ``numpy.linalg.lstsq`` (no sklearn)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_ols_beta(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Return beta (n_features,) minimizing ||X @ beta - y||_2."""
    x = np.asarray(X, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
    return beta


def predict_ols(X: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ beta
