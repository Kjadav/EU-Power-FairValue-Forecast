"""ENTSO-E Transparency Platform 
An optional folder to demonstrate an alternative data source that can be used in the pipeline,
XML parsing would be required but due to time constraint, this is not included. 
"""

from trading_pipeline_utils.entsoe_data.client import EntsoeTransparencyClient, EntsoeTransparencyError
from trading_pipeline_utils.entsoe_data.fetch import (
    aggregate_entsoe_series_to_hourly,
    fetch_actual_total_load,
    fetch_day_ahead_prices,
    fetch_entsoe_document,
)

__all__ = [
    "EntsoeTransparencyClient",
    "EntsoeTransparencyError",
    "aggregate_entsoe_series_to_hourly",
    "fetch_entsoe_document",
    "fetch_day_ahead_prices",
    "fetch_actual_total_load",
]
