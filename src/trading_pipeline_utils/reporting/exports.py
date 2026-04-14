"""Unified output persistence — pipeline run outputs under ``results/``."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trading_pipeline_utils.llm.schemas import LLMInsightResult

logger = logging.getLogger(__name__)

_SENSITIVE_JSON_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "password",
        "secret",
        "token",
        "security_token",
        "authorization",
        "refresh_token",
    }
)

def _redact_secrets(obj: Any) -> Any:
    """Return a deep copy with sensitive dict keys nulled (for config / payload snapshots)."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if str(k).lower() in _SENSITIVE_JSON_KEYS:
                out[k] = None
            else:
                out[k] = _redact_secrets(v)
        return out
    if isinstance(obj, list):
        return [_redact_secrets(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_redact_secrets(x) for x in obj)
    return obj


def create_run_directory(base_dir: Path) -> Path:
    """Create a timestamped run directory under ``results/runs/``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / "runs" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def update_latest_symlink(base_dir: Path, run_dir: Path) -> None:
    """Point ``results/latest`` at the most recent run directory."""
    latest = base_dir / "latest"
    if latest.is_symlink() or latest.exists():
        if latest.is_symlink():
            latest.unlink()
        elif latest.is_dir():
            shutil.rmtree(latest)
    try:
        latest.symlink_to(run_dir.resolve())
    except OSError:
        logger.warning("Could not create latest symlink: %s -> %s", latest, run_dir)


def save_config_snapshot(run_dir: Path, config: Any) -> Path:
    """Save the config used for this run (API keys and similar fields are stripped)."""
    path = run_dir / "config_snapshot.json"
    data = asdict(config) if hasattr(config, "__dataclass_fields__") else config
    data = _redact_secrets(data)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def save_dataframe(run_dir: Path, filename: str, df: pd.DataFrame, *, fmt: str = "csv") -> Path:
    """Save a DataFrame as CSV or Parquet."""
    path = run_dir / filename
    if fmt == "parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path, index=True)
    return path


def save_json(run_dir: Path, filename: str, data: Any) -> Path:
    """Save a dict/object as JSON (secrets redacted the same way as config snapshots)."""
    path = run_dir / filename
    if hasattr(data, "__dataclass_fields__"):
        data = asdict(data)
    data = _redact_secrets(data)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def save_text(run_dir: Path, filename: str, lines: list[str]) -> Path:
    """Save text lines to a file."""
    path = run_dir / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_run_summary(
    run_dir: Path,
    stages: dict[str, str],
    *,
    llm_insight: LLMInsightResult | None = None,
) -> Path:
    """Write a human-readable run summary."""
    lines = [
        f"Pipeline Run Summary",
        f"{'=' * 60}",
        f"Run directory: {run_dir}",
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Stage Results:",
    ]
    for stage, status in stages.items():
        lines.append(f"  {stage:30s} {status}")

    if llm_insight:
        lines.extend(["", "LLM Executive Summary:", f"  {llm_insight.executive_summary}"])

    path = run_dir / "run_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")

    json_path = run_dir / "run_summary.json"
    json_path.write_text(json.dumps({
        "run_dir": str(run_dir),
        "stages": stages,
        "llm_source": llm_insight.source if llm_insight else None,
        "executive_summary": llm_insight.executive_summary if llm_insight else None,
    }, indent=2, default=str), encoding="utf-8")

    return path
