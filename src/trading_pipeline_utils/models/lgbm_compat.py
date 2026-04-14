from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_lgbm_predict(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """Sklearn predict"""
    try:
        out = estimator.predict(X)
        return np.asarray(out, dtype=np.float64).ravel()
    except AttributeError as err:
        msg = str(err)
        if "get_params" not in msg and "super" not in msg:
            raise
    booster = getattr(estimator, "booster_", None) or getattr(estimator, "_Booster", None)
    if booster is None:
        raise AttributeError(
            "LightGBM predict failed and no booster_ was found; re-fit the production artifact "
            "or align lightgbm / scikit-learn with the training environment."
        ) from err
    try:
        names = booster.feature_name()
    except Exception:
        names = None
    if names and names != ["auto"] and all(n in X.columns for n in names):
        Xa = X.loc[:, list(names)].to_numpy(dtype=np.float64, copy=False)
    else:
        Xa = X.to_numpy(dtype=np.float64, copy=False)
    pred = booster.predict(Xa)
    return np.asarray(pred, dtype=np.float64).ravel()
