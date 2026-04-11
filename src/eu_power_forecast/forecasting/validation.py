"""Backtesting, cross-validation, and error metrics."""

from typing import Any

import numpy as np
import pandas as pd


def backtest_report(actual: pd.Series, forecast: pd.Series) -> dict[str, Any]:
    """Simple MAE / RMSE; extend with pinball loss, coverage, etc."""
    aligned = pd.concat([actual, forecast], axis=1, keys=["y", "y_hat"]).dropna()
    err = aligned["y"] - aligned["y_hat"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {"n": len(aligned), "mae": mae, "rmse": rmse}
