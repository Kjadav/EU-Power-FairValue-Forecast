from __future__ import annotations

import json
import logging
from typing import Any

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_gemini_client(api_key: str) -> Any:
    """initialise gemini client"""
    if not GENAI_AVAILABLE:
        raise ImportError("google-genai package is not installed")
    return genai.Client(api_key=api_key)


def generate_json_content(
    client: Any,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.2,
    max_output_tokens: int = 8192,
) -> str:
    """call gemini with the json file, with prompt loaded"""
    if not GENAI_AVAILABLE:
        raise ImportError("google-genai package is not installed")
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        ),
    )
    return response.text or "{}"


def strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)
    return cleaned


def repair_truncated_json(text: str) -> dict[str, Any]:
    """Parse JSON from teh response that is to be recieved
    """
    cleaned = strip_markdown_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    repaired = cleaned.rstrip()
    in_string = False
    escape = False
    for ch in repaired:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string

    if in_string:
        repaired += '"'

    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")

    if repaired.rstrip().endswith(","):
        repaired = repaired.rstrip().rstrip(",")

    repaired += "]" * max(open_brackets, 0)
    repaired += "}" * max(open_braces, 0)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    raise json.JSONDecodeError("Could not repair truncated JSON", text, 0)
