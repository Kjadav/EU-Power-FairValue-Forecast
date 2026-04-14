"""Unit tests for ENTSO-E XML parsing (no live API calls)."""

from __future__ import annotations

import textwrap

import pandas as pd
import pytest

from trading_pipeline_utils.entsoe_data.fetch import aggregate_entsoe_series_to_hourly
from trading_pipeline_utils.entsoe_data.xml_series import (
    merge_entsoe_series_frames,
    parse_publication_xml_into_series_frames,
)


def test_publication_xml_parses_points() -> None:
    xml = textwrap.dedent(
        """
        <Publication_MarketDocument xmlns="urn:test">
          <TimeSeries>
            <Period>
              <timeInterval>
                <start>2026-04-01T00:00Z</start>
                <end>2026-04-01T03:00Z</end>
              </timeInterval>
              <resolution>PT60M</resolution>
              <Point>
                <position>1</position>
                <price.amount>50.0</price.amount>
              </Point>
              <Point>
                <position>2</position>
                <price.amount>51.5</price.amount>
              </Point>
            </Period>
          </TimeSeries>
        </Publication_MarketDocument>
        """
    ).strip()
    frames = parse_publication_xml_into_series_frames(xml)
    assert len(frames) == 1
    merged = merge_entsoe_series_frames(frames, "day_ahead_price_eur_mwh")
    assert len(merged) == 2
    assert list(merged.columns) == ["day_ahead_price_eur_mwh"]
    assert merged.iloc[0, 0] == 50.0
    assert merged.index[0] == pd.Timestamp("2026-04-01 00:00:00+00:00")


def test_aggregate_15min_to_hourly_mean() -> None:
    idx = pd.date_range("2026-04-01", periods=8, freq="15min", tz="UTC")
    df = pd.DataFrame({"v": [10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0]}, index=idx)
    hourly = aggregate_entsoe_series_to_hourly(df)
    assert len(hourly) == 2
    assert hourly.iloc[0, 0] == pytest.approx(25.0)  # mean of 10..40
    assert hourly.iloc[1, 0] == pytest.approx(2.5)
