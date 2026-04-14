from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def assert_truncated_history_matches_tail(
    build_X: Callable[[pd.DataFrame], tuple[pd.DataFrame, list[str], list[str]]],
    df_full: pd.DataFrame,
    *,
    tail_start: int,
    rtol: float = 1e-9,
    atol: float = 1e-6,
    exclude_prefixes: tuple[str, ...] = ("prev_day_",),
) -> None:
    """
    leakage test to test/check if the features that are built that wont affect the future predictions
    builds based on the full history and the truncated history, the features should be dependent on that information or from previous hours

    
    """
    X_full, _num, _cat = build_X(df_full)
    df_trunc = df_full.iloc[tail_start:].copy()
    X_trunc, _, _ = build_X(df_trunc)

    idx_full = X_full.index[tail_start:]
    A = X_full.loc[idx_full]
    B = X_trunc.loc[idx_full]

    num_cols = [c for c in A.columns if str(A[c].dtype) != "category"]
    skip = {c for c in num_cols if any(c.startswith(p) for p in exclude_prefixes)}
    for c in num_cols:
        if c in skip:
            continue
        a = A[c].to_numpy(dtype=np.float64, na_value=np.nan)
        b = B[c].to_numpy(dtype=np.float64, na_value=np.nan)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.any():
            np.testing.assert_allclose(a[ok], b[ok], rtol=rtol, atol=atol, err_msg=f"column {c!r}")

    cat_cols = [c for c in A.columns if str(A[c].dtype) == "category"]
    for c in cat_cols:
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        pd.testing.assert_series_equal(A[c], B[c], check_names=True)
