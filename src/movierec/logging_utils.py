"""Small logging helpers shared by the CLI and the Streamlit app."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from contextlib import contextmanager

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s %(name)-22s %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger("movierec")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"movierec.{name}")


ProgressFn = Callable[[str, float], None]
"""Callback signature used to report pipeline progress: (message, fraction 0-1)."""


@contextmanager
def stage(logger: logging.Logger, name: str):
    """Log the start and end of a pipeline stage."""
    import time

    logger.info("→ %s", name)
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("✓ %s (%.1fs)", name, time.perf_counter() - start)
