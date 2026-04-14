import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@contextmanager
def step_timer(logger: logging.Logger, step: str, **extra: Any) -> Iterator[None]:
    start_wall = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    tail = " ".join(f"{k}={v!r}" for k, v in extra.items())
    if tail:
        logger.info("step=%s phase=start ts=%s %s", step, start_wall, tail)
    else:
        logger.info("step=%s phase=start ts=%s", step, start_wall)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info("step=%s phase=end elapsed_s=%.3f", step, elapsed)
