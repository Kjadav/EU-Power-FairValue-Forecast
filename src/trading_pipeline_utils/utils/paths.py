"""Path resolution utilities."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Repository root (three levels up from this file: src/trading_pipeline_utils/utils/)."""
    return Path(__file__).resolve().parents[3]


def default_results_dir() -> Path:
    """Default directory for pipeline run outputs (timestamped runs + ``latest`` symlink)."""
    return repo_root() / "results"


def default_data_dir() -> Path:
    """Default data directory."""
    return repo_root() / "data" / "processed"
