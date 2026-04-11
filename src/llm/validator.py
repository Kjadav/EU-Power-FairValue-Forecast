from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from data.fetcher import DataPayload
from data.validator import ValidationResult, build_profile

logger = logging.getLogger(__name__)


@dataclass
class LLMValidationResult:
    verdict: str
    confidence: float
    issues: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------

_TABLE_CONTEXT: dict[str, str] = {
    "day_ahead_prices": (
        "Hourly day-ahead wholesale electricity price for DE-LU. "
        "Values are EUR/MWh. Negative prices are normal when renewables oversupply."
    ),
    "load_forecast": "Total load / net consumption for DE-LU (MW); should be strictly positive.",
    "wind_forecast_mw": (
        "Day-ahead forecast of combined on- and offshore wind generation in DE-LU (MW)."
    ),
    "solar_forecast": "Day-ahead forecast of solar PV generation for DE-LU (MW).",
    "hydro_forecast": (
        "Day-ahead forecast for 'Sonstige' generation (MW): mainly run-of-river hydro, "
        "biomass, geothermal — not wind/PV."
    ),
    "wind_generation_actual_mw": (
        "Realized (metered) net wind generation for DE-LU (MW): on- + offshore."
    ),
    "solar_generation_actual_mw": "Realized (metered) solar PV generation for DE-LU (MW).",
    "actual_generation_total_mw": (
        "Realized net generation for DE-LU (MW): sum of all carriers."
    ),
    "hydro_pumped_storage_generation_mw": "Pumped-storage hydro generation (MW).",
}


def _build_business_context(table_name: str, columns: list[str]) -> str:
    lines = ["## Business context", ""]
    blurb = _TABLE_CONTEXT.get(table_name)
    if blurb:
        lines.append(f"**Table `{table_name}`:** {blurb}")
        lines.append("")
    lines.append("**Columns:**")
    for col in columns:
        lines.append(f"- `{col}`")
    return "\n".join(lines)


def _build_dataset_block(
    df: pd.DataFrame, table_name: str, sample_rows: int = 10
) -> str:
    columns = [str(c) for c in df.columns]
    sections = [
        _build_business_context(table_name, columns),
        "", "---", "",
        f"dataset_id: {table_name}",
        f"n_rows: {len(df)}",
        f"columns: {columns}",
        f"index_type: {type(df.index).__name__}",
    ]
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        sections.append(f"index_tz: {idx.tz}")
        sections.append(f"index_range: {idx.min()} .. {idx.max()}")
    sections.append(
        f"dtypes: {json.dumps({c: str(df[c].dtype) for c in df.columns})}"
    )
    sections.append("sample_csv:\n" + df.head(sample_rows).to_csv())
    profile = build_profile(df, table_name)
    sections.append("profile_json:\n" + json.dumps(profile, indent=2, default=str))
    return "\n".join(sections)


def _load_system_prompt(config: dict[str, Any]) -> str:
    path = Path(config.get("llm", {}).get("prompt_path", "prompts/quality_prompt.md"))
    if not path.is_file():
        raise FileNotFoundError(f"LLM QA prompt not found: {path}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# rule parsing
# ---------------------------------------------------------------------------

_ALLOWED_RULE_TYPES = frozenset({
    "not_null", "range", "frequency_check", "no_missing_timestamps",
    "monotonic_time_index", "outlier", "volatility_jump",
})

_RANGE_RE = re.compile(r"^\s*(>=|<=|>|<)\s*([-+eE0-9.]+)\s*$")


def _validate_rule(rule: dict[str, Any]) -> None:
    required = {"rule_name", "type", "column", "condition", "confidence", "reasoning"}
    missing = required - set(rule.keys())
    if missing:
        raise ValueError(f"rule missing keys {sorted(missing)}")
    if rule["type"] not in _ALLOWED_RULE_TYPES:
        raise ValueError(f"unsupported rule type {rule['type']!r}")


def _parse_rules(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    if isinstance(payload, list):
        rules = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rules"), list):
        rules = payload["rules"]
    else:
        raise ValueError("expected JSON with 'rules' array")
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("each rule must be an object")
        _validate_rule(rule)
    return rules


# ---------------------------------------------------------------------------
# rule runners
# ---------------------------------------------------------------------------

def _cond_map(condition: Any) -> dict[str, Any]:
    if condition is None or condition == "":
        return {}
    if isinstance(condition, dict):
        return condition
    if isinstance(condition, str):
        try:
            return json.loads(condition)
        except json.JSONDecodeError:
            return {}
    return {}


def _require_dt_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("expected DatetimeIndex")
    return df.index


def _run_not_null(df: pd.DataFrame, column: str, condition: Any) -> dict[str, Any]:
    if column == "index" or column not in df.columns:
        return {"ok": False, "error": f"invalid column {column}"}
    m = _cond_map(condition)
    max_null = float(m.get("max_null_fraction", 0.0))
    null_frac = float(df[column].isna().mean())
    return {"ok": null_frac <= max_null, "null_fraction": null_frac}


def _run_range(df: pd.DataFrame, column: str, condition: Any) -> dict[str, Any]:
    if column not in df.columns:
        return {"ok": False, "error": f"missing column {column}"}
    if not isinstance(condition, str):
        raise ValueError("range condition must be a string")
    match = _RANGE_RE.match(condition.strip())
    if not match:
        raise ValueError(f"bad range condition: {condition!r}")
    op, thresh = match.group(1), float(match.group(2))
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    ops = {
        ">=": values < thresh,
        "<=": values > thresh,
        ">": values <= thresh,
        "<": values >= thresh,
    }
    violations = int(ops[op].sum())
    return {"ok": violations == 0, "violations": violations}


def _run_frequency(df: pd.DataFrame, column: str, condition: Any) -> dict[str, Any]:
    if column != "index":
        return {"ok": False, "error": "frequency_check requires column == 'index'"}
    idx = _require_dt_index(df)
    m = _cond_map(condition)
    expected = float(m.get("expected_step_seconds", 3600))
    tol = float(m.get("tolerance_seconds", 120.0))
    min_frac = float(m.get("min_fraction_within_tolerance", 0.95))
    if len(idx) < 2:
        return {"ok": True, "note": "too_few_points"}
    deltas = idx.to_series().diff().dropna().dt.total_seconds()
    within = ((deltas - expected).abs() <= tol).mean()
    return {"ok": float(within) >= min_frac, "fraction_within_tolerance": float(within)}


def _run_no_missing(df: pd.DataFrame, column: str, condition: Any) -> dict[str, Any]:
    if column != "index":
        return {"ok": False, "error": "no_missing_timestamps requires column == 'index'"}
    idx = _require_dt_index(df)
    m = _cond_map(condition)
    max_gap = float(m.get("max_gap_seconds", 7200))
    if len(idx) < 2:
        return {"ok": True, "note": "too_few_points"}
    deltas = idx.to_series().diff().dropna().dt.total_seconds()
    bad = int((deltas > max_gap).sum())
    return {"ok": bad == 0, "large_gap_count": bad}


def _run_monotonic(df: pd.DataFrame, column: str, condition: Any) -> dict[str, Any]:
    if column != "index":
        return {"ok": False, "error": "monotonic_time_index requires column == 'index'"}
    idx = _require_dt_index(df)
    dupes = int(idx.duplicated().sum())
    mono = bool(idx.is_monotonic_increasing)
    return {"ok": mono and dupes == 0, "is_monotonic": mono, "duplicate_index_count": dupes}


def _run_outlier(df: pd.DataFrame, column: str, condition: Any) -> dict[str, Any]:
    if column == "index" or column not in df.columns:
        return {"ok": False, "error": "outlier needs an existing value column"}
    m = _cond_map(condition)
    z_thresh = float(m.get("z_threshold", 4.0))
    max_frac = float(m.get("max_outlier_fraction", 0.05))
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(series) < 10:
        return {"ok": True, "note": "too_few_points"}
    mean, std = float(series.mean()), float(series.std(ddof=0))
    if std == 0.0:
        return {"ok": True, "note": "constant_series"}
    frac = float(((series - mean) / std).abs().gt(z_thresh).mean())
    return {"ok": frac <= max_frac, "outlier_fraction": frac}


def _run_volatility(df: pd.DataFrame, column: str, condition: Any) -> dict[str, Any]:
    if column == "index" or column not in df.columns:
        return {"ok": False, "error": "volatility_jump needs an existing value column"}
    m = _cond_map(condition)
    cap = float(m.get("max_abs_change", 1e12))
    step = pd.to_numeric(df[column], errors="coerce").diff().abs().max()
    if pd.isna(step):
        return {"ok": True, "note": "no_pairs"}
    return {"ok": float(step) <= cap, "max_abs_step_observed": float(step)}


_RUNNERS: dict[str, Callable[[pd.DataFrame, str, Any], dict[str, Any]]] = {
    "not_null": _run_not_null,
    "range": _run_range,
    "frequency_check": _run_frequency,
    "no_missing_timestamps": _run_no_missing,
    "monotonic_time_index": _run_monotonic,
    "outlier": _run_outlier,
    "volatility_jump": _run_volatility,
}


def _execute_rules(
    df: pd.DataFrame,
    rules: list[dict[str, Any]],
    min_confidence: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    enforced_failures = 0
    for rule in rules:
        runner = _RUNNERS.get(rule["type"])
        if runner is None:
            detail: dict[str, Any] = {"ok": False, "error": f"unknown type {rule['type']!r}"}
        else:
            try:
                detail = runner(df, rule["column"], rule["condition"])
            except Exception as exc:
                detail = {"ok": False, "error": str(exc)}
        enforced = rule["confidence"] >= min_confidence
        passed = bool(detail.get("ok", False))
        if enforced and not passed:
            enforced_failures += 1
        results.append({
            "rule_name": rule["rule_name"],
            "type": rule["type"],
            "column": rule["column"],
            "confidence": rule["confidence"],
            "enforced": enforced,
            "passed": passed,
            "detail": detail,
        })
    return {
        "ok": enforced_failures == 0,
        "enforced_failures": enforced_failures,
        "results": results,
    }


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _call_llm(
    table_name: str,
    dataset_block: str,
    system_prompt: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    from openai import OpenAI

    llm_cfg = config.get("llm", {})
    api_key = llm_cfg.get("api_key")
    if not api_key:
        raise ValueError("LLM API key not configured")
    client = OpenAI(api_key=api_key)
    user_msg = (
        f"Table: {table_name}\n\n{dataset_block}\n\n"
        "Return JSON with key `rules` only, per system instructions."
    )
    response = client.chat.completions.create(
        model=llm_cfg.get("model", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=llm_cfg.get("temperature", 0.2),
    )
    content = response.choices[0].message.content or "{}"
    return _parse_rules(content)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def validate(
    data: DataPayload,
    validation_result: ValidationResult,
    config: dict[str, Any],
) -> LLMValidationResult:
    """Run LLM-assisted validation on each core table."""
    llm_cfg = config.get("llm", {})
    min_conf = llm_cfg.get("min_confidence", 0.85)
    system_prompt = _load_system_prompt(config)

    all_issues: list[dict[str, Any]] = []
    table_details: dict[str, Any] = {}
    all_ok = True

    for table_name, df in data.iter_core_tables():
        if df.empty:
            table_details[table_name] = {"skipped": "empty"}
            continue
        try:
            dataset_block = _build_dataset_block(df, table_name)
            rules = _call_llm(table_name, dataset_block, system_prompt, config)
            execution = _execute_rules(df, rules, min_conf)
            table_details[table_name] = {
                "rules_proposed": len(rules),
                "execution": execution,
            }
            if not execution["ok"]:
                all_ok = False
                for r in execution["results"]:
                    if r["enforced"] and not r["passed"]:
                        all_issues.append({
                            "table": table_name,
                            "rule": r["rule_name"],
                            "detail": r["detail"],
                        })
        except Exception as exc:
            logger.warning("LLM validation failed for %s: %s", table_name, exc)
            table_details[table_name] = {"error": str(exc)}
            all_ok = False
            all_issues.append({"table": table_name, "error": str(exc)})

    if all_ok:
        verdict = "pass"
    elif all_issues:
        verdict = "fail"
    else:
        verdict = "review"

    stat_ok = validation_result.ok
    combined_ok = stat_ok and all_ok
    confidence = 1.0 if combined_ok else 0.5

    return LLMValidationResult(
        verdict=verdict,
        confidence=confidence,
        issues=all_issues,
        details={"tables": table_details, "statistical_ok": stat_ok},
    )
