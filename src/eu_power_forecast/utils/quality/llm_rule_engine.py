"""Validate LLM JSON, parse rules, and execute them on dataframes."""

from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd

from eu_power_forecast.utils.quality.llm_rule_runners import (
    run_frequency_check_rule,
    run_monotonic_time_index_rule,
    run_no_missing_timestamps_rule,
    run_not_null_rule,
    run_outlier_rule,
    run_range_rule,
    run_volatility_jump_rule,
)

ALLOWED_LLM_RULE_TYPES = frozenset(
    {
        "not_null",
        "range",
        "frequency_check",
        "no_missing_timestamps",
        "monotonic_time_index",
        "outlier",
        "volatility_jump",
    }
)

RuleRunner = Callable[[pd.DataFrame, str, Any], dict[str, Any]]

RULE_RUNNERS_BY_TYPE: dict[str, RuleRunner] = {
    "not_null": run_not_null_rule,
    "range": run_range_rule,
    "frequency_check": run_frequency_check_rule,
    "no_missing_timestamps": run_no_missing_timestamps_rule,
    "monotonic_time_index": run_monotonic_time_index_rule,
    "outlier": run_outlier_rule,
    "volatility_jump": run_volatility_jump_rule,
}


def validate_llm_rule_structure(rule: dict[str, Any]) -> None:
    required = {"rule_name", "type", "column", "condition", "confidence", "reasoning"}
    missing = required - set(rule.keys())
    if missing:
        raise ValueError(f"rule missing keys {sorted(missing)}: {rule}")
    if rule["type"] not in ALLOWED_LLM_RULE_TYPES:
        raise ValueError(f"unsupported rule type {rule['type']!r}")


def parse_llm_rules_json_response(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    if isinstance(payload, list):
        rules = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rules"), list):
        rules = payload["rules"]
    else:
        raise ValueError("expected JSON object with 'rules' array or a JSON array of rules")
    if not isinstance(rules, list):
        raise ValueError("'rules' must be a list")
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("each rule must be an object")
        validate_llm_rule_structure(rule)
    return rules


def execute_llm_rules(
    df: pd.DataFrame,
    rules: list[dict[str, Any]],
    *,
    min_confidence_to_enforce: float = 0.85,
) -> dict[str, Any]:
    """Run rules; low-confidence rules are recorded but not enforced as failures."""
    results: list[dict[str, Any]] = []
    enforced_failures = 0
    for rule in rules:
        rule_type = str(rule["type"])
        column = str(rule["column"])
        confidence = float(rule["confidence"])
        runner = RULE_RUNNERS_BY_TYPE.get(rule_type)
        if runner is None:
            detail = {"ok": False, "error": f"unknown rule type {rule_type!r}"}
        else:
            try:
                detail = runner(df, column, rule["condition"])
            except Exception as exc:  # noqa: BLE001
                detail = {"ok": False, "error": str(exc)}
        enforced = confidence >= min_confidence_to_enforce
        passed = bool(detail.get("ok", False))
        if enforced and not passed:
            enforced_failures += 1
        results.append(
            {
                "rule_name": rule["rule_name"],
                "type": rule_type,
                "column": column,
                "confidence": confidence,
                "reasoning": rule["reasoning"],
                "enforced": enforced,
                "passed": passed,
                "detail": detail,
            }
        )
    return {
        "ok": enforced_failures == 0,
        "enforced_failures": enforced_failures,
        "rules_evaluated": len(results),
        "results": results,
    }
