"""Walk-forward day-ahead price regression backtest."""

__all__ = [
    "deploy_regression_backtest",
    "walk_forward_day_ahead_backtest",
]


def __getattr__(name: str):
    if name == "walk_forward_day_ahead_backtest":
        from backtest.regression.walkforward import walk_forward_day_ahead_backtest as _wf

        return _wf
    if name == "deploy_regression_backtest":
        from backtest.regression.pipeline import deploy_regression_backtest as _dep

        return _dep
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
