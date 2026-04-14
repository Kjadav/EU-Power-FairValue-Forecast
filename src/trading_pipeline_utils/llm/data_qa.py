from __future__ import annotations

import json
import logging

import pandas as pd

from trading_pipeline_utils.data.schemas import (
    DataPayload,
    SmardData,
    ValidationResult,
    build_table_context_markdown,
)
from trading_pipeline_utils.llm.client import create_gemini_client, generate_json_content, strip_markdown_fences, GENAI_AVAILABLE
from trading_pipeline_utils.llm.prompts import DATA_QA_SYSTEM_PROMPT, DATA_QA_USER_TEMPLATE
from trading_pipeline_utils.llm.schemas import LLMValidationResult
from trading_pipeline_utils.settings import PipelineConfig

logger = logging.getLogger(__name__)


def _build_bundle_digest(bundle: SmardData) -> str:
    parts: list[str] = ["## Bundle digest (core tables)", ""]
    for name, df in bundle.iter_core_tables():
        parts.append(build_table_context_markdown(name, [str(c) for c in df.columns]))
        parts.append(f"rows={len(df)} cols={list(df.columns)}")
        if len(df) and isinstance(df.index, pd.DatetimeIndex):
            parts.append(f"index_range={df.index.min()} .. {df.index.max()}")
        parts.append("")
    return "\n".join(parts)


def _build_user_message(digest: str, validation_json: str) -> str:
    """Render the prompt from the template."""
    return DATA_QA_USER_TEMPLATE.substitute(
        bundle_digest=digest,
        validation_json=validation_json,
    )


def _parse_llm_json(text: str) -> LLMValidationResult:
    """Parse Gemini's JSON response into am result that is returned"""
    data = json.loads(text)
    v = str(data.get("verdict", "review")).lower()
    if v not in ("pass", "fail", "review"):
        v = "review"
    conf = float(data.get("confidence", 0.0))
    conf = max(0.0, min(1.0, conf))
    issues_raw = data.get("issues") or []
    issues: list[dict[str, str]] = []
    if isinstance(issues_raw, list):
        for item in issues_raw:
            if isinstance(item, dict):
                issues.append(
                    {
                        "code": str(item.get("code", "unknown")),
                        "detail": str(item.get("detail", "")),
                    }
                )
    return LLMValidationResult(verdict=v, confidence=conf, issues=issues, raw_response=text)


def validate(
    data: DataPayload,
    validation_result: ValidationResult,
    config: PipelineConfig,
) -> LLMValidationResult:
    """Run LLM-based data quality validation on a SMARD data that is returned

    """
    key = config.llm.api_key
    if not key:
        return LLMValidationResult(
            verdict="review",
            confidence=0.0,
            issues=[{"code": "llm_skipped", "detail": "LLM API key not configured"}],
        )

    if not GENAI_AVAILABLE:
        return LLMValidationResult(
            verdict="review",
            confidence=0.0,
            issues=[{"code": "llm_skipped", "detail": "google-genai package not installed"}],
        )

    digest = _build_bundle_digest(data.bundle)
    validation_json = json.dumps(
        {"ok": validation_result.ok, "summary": validation_result.summary},
        indent=2,
        default=str,
    )
    user_message = _build_user_message(digest, validation_json)
    full_prompt = f"{DATA_QA_SYSTEM_PROMPT}\n\n{user_message}"

    client = create_gemini_client(key)
    try:
        raw_text = generate_json_content(
            client,
            config.llm.model,
            full_prompt,
            temperature=config.llm.temperature,
            max_output_tokens=config.llm.max_tokens,
        )
        cleaned = strip_markdown_fences(raw_text)
        return _parse_llm_json(cleaned)
    except Exception as e:
        logger.warning("LLM validation failed: %s", e)
        return LLMValidationResult(
            verdict="review",
            confidence=0.0,
            issues=[{"code": "llm_error", "detail": str(e)}],
            error=str(e),
        )
