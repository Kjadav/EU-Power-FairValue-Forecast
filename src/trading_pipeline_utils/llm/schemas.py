from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["pass", "fail", "review"]

INSIGHT_FIELDS = [
    "executive_summary", "outright_views", "shape_views", "why_now",
    "what_the_desk_would_do", "key_risks", "invalidation_triggers",
    "what_would_change_the_view", "confidence_notes", "data_quality_notes",
    "questions_for_trader",
]


@dataclass(frozen=True)
class LLMValidationResult:
    verdict: Verdict
    confidence: float
    issues: list[dict[str, str]]
    raw_response: str | None = None
    error: str | None = None


@dataclass
class LLMInsightResult:
    executive_summary: str = ""
    outright_views: list[str] = field(default_factory=list)
    shape_views: list[str] = field(default_factory=list)
    why_now: str = ""
    what_the_desk_would_do: str = ""
    key_risks: list[str] = field(default_factory=list)
    invalidation_triggers: list[str] = field(default_factory=list)
    what_would_change_the_view: str = ""
    confidence_notes: str = ""
    data_quality_notes: str = ""
    questions_for_trader: list[str] = field(default_factory=list)
    source: str = "deterministic_fallback"
    error: str | None = None
