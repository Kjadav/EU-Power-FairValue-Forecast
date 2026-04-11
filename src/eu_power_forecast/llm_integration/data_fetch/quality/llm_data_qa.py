"""
Post-fetch table QA: optional OpenAI rule proposals, then deterministic rule execution.

Logic lives in ``eu_power_forecast.utils.quality``; this module wires OpenAI and SMARD bundles.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from eu_power_forecast.llm_integration.data_fetch.quality.dataset_payload import build_llm_dataset_text_block
from eu_power_forecast.utils.quality import (
    ALLOWED_LLM_RULE_TYPES,
    build_dataframe_profile,
    execute_llm_rules,
    parse_llm_rules_json_response,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent / "quality_prompt.md"
if not _PROMPT_PATH.is_file():
    raise FileNotFoundError(f"missing LLM QA prompt: {_PROMPT_PATH}")

QUALITY_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

if TYPE_CHECKING:
    from eu_power_forecast.ingestion.smard_data import SmardData


def propose_rules_openai(
    table_name: str,
    dataset_text_block: str,
    *,
    model: str = "gpt-4o-mini",
    client: Any = None,
) -> list[dict[str, Any]]:
    """Call OpenAI chat completions; returns parsed ``rules`` list."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    openai_client = client or OpenAI(api_key=api_key)
    user_message = (
        f"Table: {table_name}\n\n{dataset_text_block}\n\n"
        "Return JSON with key `rules` only, per system instructions."
    )
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": QUALITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = response.choices[0].message.content or "{}"
    return parse_llm_rules_json_response(content)


def qa_single_table(
    dataframe: pd.DataFrame,
    table_name: str,
    *,
    use_llm: bool = False,
    model: str = "gpt-4o-mini",
    client: Any = None,
    sample_rows: int = 10,
    min_confidence_to_enforce: float = 0.85,
) -> dict[str, Any]:
    """Profile the table; optionally fetch LLM rules and execute them on the full frame."""
    profile = build_dataframe_profile(dataframe, table_name)
    duplicate_index_ok = profile["duplicate_index_count"] == 0
    report: dict[str, Any] = {
        "profile": profile,
        "llm_validation": None,
        "min_confidence_to_enforce": min_confidence_to_enforce,
    }

    if not use_llm:
        report["overall_ok"] = duplicate_index_ok
        report["summary"] = {
            "profile_ok": duplicate_index_ok,
            "overall_ok": duplicate_index_ok,
            "note": "llm_skipped",
        }
        return report

    try:
        rules = propose_rules_openai(
            table_name,
            build_llm_dataset_text_block(dataframe, table_name, sample_row_count=sample_rows),
            model=model,
            client=client,
        )
        execution = execute_llm_rules(
            dataframe,
            rules,
            min_confidence_to_enforce=min_confidence_to_enforce,
        )
        rule_results = execution["results"]
        enforced_rows = [row for row in rule_results if row.get("enforced")]
        all_enforced_passed = all(row.get("passed") for row in enforced_rows) if enforced_rows else True
        report["llm_validation"] = {
            "rules_proposed": rules,
            "rules_executed": rule_results,
            "all_enforced_passed": all_enforced_passed,
            "execution_summary": {
                "ok": execution["ok"],
                "enforced_failures": execution["enforced_failures"],
                "rules_evaluated": execution["rules_evaluated"],
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM QA failed %s: %s", table_name, exc)
        report["llm_validation"] = {"error": str(exc)}

    llm_block = report["llm_validation"] or {}
    if llm_block.get("error"):
        report["overall_ok"] = False
        report["summary"] = {
            "profile_ok": duplicate_index_ok,
            "overall_ok": False,
            "llm_error": llm_block["error"],
        }
    else:
        executed_rows = llm_block.get("rules_executed") or []
        enforced_rows = [row for row in executed_rows if row.get("enforced")]
        rules_pass = all(row.get("passed") for row in enforced_rows) if enforced_rows else True
        report["overall_ok"] = duplicate_index_ok and rules_pass
        report["summary"] = {
            "profile_ok": duplicate_index_ok,
            "rules_n": len(executed_rows),
            "enforced_n": len(enforced_rows),
            "failed_rule_names": [row["rule_name"] for row in enforced_rows if not row.get("passed")],
            "overall_ok": report["overall_ok"],
        }
    return report


def qa_smard_bundle_report(
    data: SmardData,
    *,
    use_llm: bool = False,
    model: str = "gpt-4o-mini",
    client: Any = None,
    min_confidence_to_enforce: float = 0.85,
) -> dict[str, Any]:
    """Run ``qa_single_table`` on each core DE-LU hourly table in a ``SmardData`` bundle."""
    from eu_power_forecast.ingestion.smard_data import SmardData as SmardDataType

    if not isinstance(data, SmardDataType):
        raise TypeError("expected SmardData")

    tables = dict(data.iter_core_tables())
    bundle_report: dict[str, Any] = {
        "region": data.region,
        "resolution": data.resolution,
        "rule_taxonomy": sorted(ALLOWED_LLM_RULE_TYPES),
        "min_confidence_to_enforce": min_confidence_to_enforce,
        "tables": {},
    }
    all_tables_ok = True
    for table_name, table_df in tables.items():
        table_report = qa_single_table(
            table_df,
            table_name,
            use_llm=use_llm,
            model=model,
            client=client,
            min_confidence_to_enforce=min_confidence_to_enforce,
        )
        bundle_report["tables"][table_name] = table_report
        all_tables_ok = all_tables_ok and bool(table_report.get("overall_ok"))
    bundle_report["summary"] = {"overall_ok": all_tables_ok, "llm_enabled": use_llm}
    return bundle_report


# Backward-compatible names for imports that expect the old ingestion API
dataframe_profile = build_dataframe_profile
baseline_qa_dataframe = build_dataframe_profile
parse_llm_rules_response = parse_llm_rules_json_response
