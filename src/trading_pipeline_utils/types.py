from typing import Literal

AggregationLevel = Literal["scenario_paths", "quantiles_approx", "point_only"]
Signal = Literal["LONG", "SHORT", "NEUTRAL"]
Outlook = Literal["Bullish", "Bearish", "Uncertain"]
Verdict = Literal["pass", "fail", "review"]
