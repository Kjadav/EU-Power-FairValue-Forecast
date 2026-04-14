from __future__ import annotations

import os
ENV_ENTSOE_TOKEN = "ENTSOE_TOKEN"
ENV_ENTSOE_API_BASE = "ENTSOE_API_BASE"
DEFAULT_API_BASE = "https://web-api.tp.entsoe.eu/api"
DEFAULT_BIDDING_ZONE_EIC = "10Y1001A1001A82H"


def entsoe_token() -> str:
    """Return the API security token from the environment."""
    t = os.environ.get(ENV_ENTSOE_TOKEN, "").strip()
    if not t:
        raise ValueError(
            f"Missing {ENV_ENTSOE_TOKEN}. "
        )
    return t


def api_base_url() -> str:
    return os.environ.get(ENV_ENTSOE_API_BASE, DEFAULT_API_BASE).rstrip("/")
