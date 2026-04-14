"""Thin repo-root entrypoint — delegates to the installed pipeline CLI."""

from __future__ import annotations

from trading_pipeline_utils.main import run_pipeline

if __name__ == "__main__":
    raise SystemExit(run_pipeline())
