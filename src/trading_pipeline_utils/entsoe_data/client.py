from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlencode

import requests

from trading_pipeline_utils.entsoe_data.config import api_base_url, entsoe_token

logger = logging.getLogger(__name__)


class EntsoeTransparencyError(RuntimeError):
    """Raised when ENTSO-E does not give us a usable document.
    """


class EntsoeTransparencyClient:
    def __init__(
        self,
        *,
        token: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.token = token if token is not None else entsoe_token()
        self.base_url = (base_url or api_base_url()).rstrip("/")
        self.timeout_s = timeout_s
