"""AI-assisted steps: summarization, spec-to-code, report drafting, etc."""

from pathlib import Path
from typing import Any


def run_assisted_step(
    prompt_name: str,
    prompts_dir: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a prompt template from prompts/ and call your provider; stub for now."""
    _ = (prompt_name, prompts_dir, kwargs)
    return {"status": "stub", "message": "Plug in LLM client and prompt rendering"}
