"""Regression backtest is **not** executed from this module.

After each ingest, ``scripts/run_ingestion.py`` runs ``deploy_regression_backtest`` on the bundle
and writes ``data/processed/smard_bundle/regression_backtest.json`` (metrics only; no large frames).

For interactive charts and prediction diagrams, open ``notebooks/regression_backtesting.ipynb``.

Ad-hoc CLI (repo root, ``PYTHONPATH`` including ``src`` and ``.``)::

    python -c "from backtest.regression.pipeline import deploy_regression_backtest, default_bundle_dir; \\
        deploy_regression_backtest(default_bundle_dir(), json_path=default_bundle_dir().parent / 'regression_cli.json')"
"""
