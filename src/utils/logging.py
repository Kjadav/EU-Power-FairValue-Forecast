import logging
import time
from contextlib import contextmanager
from typing import Any, Generator


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@contextmanager
def step_timer(
    step_name: str, logger: logging.Logger | None = None
) -> Generator[dict[str, Any], None, None]:
    log = logger or logging.getLogger("pipeline")
    log.info("[%s] started", step_name)
    t0 = time.monotonic()
    context: dict[str, Any] = {"step": step_name, "start": time.time()}
    try:
        yield context
    finally:
        elapsed = time.monotonic() - t0
        context["elapsed_s"] = round(elapsed, 3)
        log.info("[%s] completed in %.2fs", step_name, elapsed)
