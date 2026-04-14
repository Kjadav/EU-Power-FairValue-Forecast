from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from trading_pipeline_utils.entsoe_data.client import EntsoeTransparencyClient
from trading_pipeline_utils.entsoe_data.config import DEFAULT_BIDDING_ZONE_EIC
from trading_pipeline_utils.entsoe_data.xml_series import (
    merge_entsoe_series_frames,
    parse_publication_xml_into_series_frames,
)

logger = logging.getLogger(__name__)


DOC_PRICE = "A44"  # day-ahead auction results
DOC_ACTUAL_LOAD = "A65"  # Actual total load 
PROCESS_TYPE_REALISED = "A16"  # realised


def aggregate_entsoe_series_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_index()
    out = out.resample("1h", label="left", closed="left").mean()
    return out.dropna(how="all")


def period_params(start: datetime, end: datetime) -> dict[str, str]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware (use UTC)")
    su = start.astimezone(timezone.utc)
    eu = end.astimezone(timezone.utc)
    return {
        "periodStart": su.strftime("%Y%m%d%H%M"),
        "periodEnd": eu.strftime("%Y%m%d%H%M"),
    }


def fetch_entsoe_document(
    document_type: str,
    *,
    client: EntsoeTransparencyClient | None = None,
    extra_params: dict[str, Any] | None = None,
) -> str:
    c = client or EntsoeTransparencyClient()
    params: dict[str, Any] = dict(extra_params or {})
    params.setdefault("documentType", document_type)
    return c.fetch_xml(params)


def fetch_day_ahead_prices(
    start: datetime,
    end: datetime,
    *,
    in_domain: str = DEFAULT_BIDDING_ZONE_EIC,
    out_domain: str = DEFAULT_BIDDING_ZONE_EIC,
    value_column: str = "day_ahead_price_eur_mwh",
    client: EntsoeTransparencyClient | None = None,
    aggregate_to_hourly: bool = True,
) -> pd.DataFrame:
    params = period_params(start, end)
    params.update(
        {
            "in_Domain": in_domain,
            "out_Domain": out_domain,
        }
    )
    xml_text = fetch_entsoe_document(DOC_PRICE, client=client, extra_params=params)
    frames = parse_publication_xml_into_series_frames(xml_text)
    df = merge_entsoe_series_frames(frames, value_column)
    if df.empty:
        logger.warning("ENTSO-E A44 returned no parseable TimeSeries for %s–%s", start, end)
    elif aggregate_to_hourly:
        df = aggregate_entsoe_series_to_hourly(df)
    return df


def fetch_actual_total_load(
    start: datetime,
    end: datetime,
    *,
    bidding_zone: str = DEFAULT_BIDDING_ZONE_EIC,
    value_column: str = "total_load_mw",
    client: EntsoeTransparencyClient | None = None,
    aggregate_to_hourly: bool = True,
) -> pd.DataFrame:
    params = period_params(start, end)
    params.update(
        {
            "processType": PROCESS_TYPE_REALISED,
            "outBiddingZone_Domain": bidding_zone,
        }
    )
    xml_text = fetch_entsoe_document(DOC_ACTUAL_LOAD, client=client, extra_params=params)
    frames = parse_publication_xml_into_series_frames(xml_text)
    df = merge_entsoe_series_frames(frames, value_column)
    if df.empty:
        logger.warning("ENTSO-E A65 returned no parseable TimeSeries for %s–%s", start, end)
    elif aggregate_to_hourly:
        df = aggregate_entsoe_series_to_hourly(df)
    return df
