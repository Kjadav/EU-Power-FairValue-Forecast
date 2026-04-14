"""Run the full trading pipeline from the repo root.

    python scripts/run_pipeline.py

Config is loaded via :func:`trading_pipeline_utils.settings.load_config` (default YAML
next to the package). Use ``trading-pipeline --help`` for the CLI name only.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(_REPO_ROOT / "src"))

from trading_pipeline_utils.main import run_pipeline

if __name__ == "__main__":
    raise SystemExit(run_pipeline())
