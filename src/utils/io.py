import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_env_secrets(config: dict[str, Any]) -> dict[str, Any]:
    vendor_key = os.environ.get("DATA_VENDOR_API_KEY")
    if vendor_key:
        config.setdefault("data_vendor", {})["api_key"] = vendor_key
    llm_key = os.environ.get("LLM_API_KEY")
    if llm_key:
        config.setdefault("llm", {})["api_key"] = llm_key
    return config
