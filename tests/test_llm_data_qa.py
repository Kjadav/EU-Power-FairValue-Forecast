import pandas as pd
import pytest

from data.validator import build_profile
from llm.validator import _execute_rules, _parse_rules


def test_dataframe_profile():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({"x": [1.0, float("nan"), 3.0, 4.0, 5.0]}, index=idx)
    b = build_profile(df, "t")
    assert b["n_rows"] == 5
    assert b["missingness"]["x"] == pytest.approx(0.2)
    assert b["time_coverage"]["median_step_seconds"] == 3600.0


def _r(name: str, typ: str, col: str, cond, conf: float = 0.95) -> dict:
    return {
        "rule_name": name,
        "type": typ,
        "column": col,
        "condition": cond,
        "confidence": conf,
        "reasoning": "test",
    }


def _first_result(execution: dict) -> dict:
    return execution["results"][0]


def test_range_and_not_null():
    df = pd.DataFrame({"amount": [0.0, 1.0, 2.0]})
    out = _execute_rules(df, [_r("rng", "range", "amount", ">= 0")], 0.85)
    assert _first_result(out)["passed"] is True

    df2 = pd.DataFrame({"amount": [0.0, None, 2.0]})
    out2 = _execute_rules(
        df2,
        [_r("nn", "not_null", "amount", {"max_null_fraction": 0.5})],
        0.85,
    )
    assert _first_result(out2)["passed"] is True


def test_range_fails():
    df = pd.DataFrame({"p": [10.0, -50.0, 20.0]})
    out = _execute_rules(df, [_r("m", "range", "p", ">= 0")], 0.85)
    assert _first_result(out)["passed"] is False


def test_monotonic_index():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({"x": range(5)}, index=idx)
    out = _execute_rules(df, [_r("mono", "monotonic_time_index", "index", {})], 0.85)
    assert _first_result(out)["passed"] is True


def test_low_confidence_not_enforced():
    df = pd.DataFrame({"amount": [-5.0]})
    out = _execute_rules(
        df,
        [_r("x", "range", "amount", ">= 0", conf=0.3)],
        0.85,
    )
    assert _first_result(out)["enforced"] is False


def test_parse_rules_response():
    text = '{"rules": [{"rule_name":"a","type":"range","column":"c","condition":">= 0","confidence":0.9,"reasoning":"x"}]}'
    rules = _parse_rules(text)
    assert len(rules) == 1
    assert rules[0]["rule_name"] == "a"


def test_parse_rules_array():
    rules = _parse_rules(
        '[{"rule_name":"z","type":"range","column":"c","condition":">= 0","confidence":0.9,"reasoning":"x"}]'
    )
    assert len(rules) == 1


def test_quality_prompt_loads():
    from pathlib import Path

    prompt_path = Path("prompts/quality_prompt.md")
    assert prompt_path.is_file(), f"prompt file missing: {prompt_path}"
    content = prompt_path.read_text(encoding="utf-8")
    assert "frequency_check" in content
    assert "confidence" in content
