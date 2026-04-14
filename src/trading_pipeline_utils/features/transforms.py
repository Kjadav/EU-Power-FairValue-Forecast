from __future__ import annotations

import numpy as np
import pandas as pd


class ArcsinhPriceTransform:
    """
    Transforming price using arcsinh to combat negative pricing
    y - arcsinh(price / scale); scale = max(floor, median(|price|)) 
    """

    def __init__(self, scale_floor: float = 10.0) -> None:
        self.scale_floor = float(scale_floor)
        self.scale_: float | None = None

    def fit(self, price_train: pd.Series | np.ndarray) -> ArcsinhPriceTransform:
        s = np.asarray(price_train, dtype=np.float64)
        med = float(np.median(np.abs(s[np.isfinite(s)])))
        self.scale_ = max(self.scale_floor, med)
        return self

    def transform(self, price: pd.Series | np.ndarray) -> np.ndarray:
        if self.scale_ is None:
            raise RuntimeError("Call fit() before transform()")
        x = np.asarray(price, dtype=np.float64) / self.scale_
        return np.arcsinh(x)

    def inverse_transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        if self.scale_ is None:
            raise RuntimeError("Call fit() before inverse_transform()")
        yy = np.asarray(y, dtype=np.float64)
        return np.sinh(yy) * self.scale_
