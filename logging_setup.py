"""Loguru-based logging setup.

NOTE: This file is intentionally named `logging_setup.py` rather than `logging.py`
to avoid shadowing the Python standard-library `logging` module on `sys.path`.
The master plan calls it `logging.py`; the rename is a safety-only deviation
documented in HANDOFF.md (gotcha #1).

Usage:
    from logging_setup import init_logging
    init_logging(run_id="20260514-1500")
    from loguru import logger
    logger.info("...")
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def init_logging(
    run_id: str,
    level: str = "INFO",
    rotation_mb: int = 50,
    runs_root: str = "runs",
) -> Path:
    """Initialize loguru sinks: stderr + per-run log file.

    Returns the run directory path.
    """
    run_dir = Path(runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()  # drop default stderr sink
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        run_dir / "log.txt",
        level=level,
        rotation=f"{rotation_mb} MB",
        retention=10,
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    logger.info(f"Logging initialized for run_id={run_id} -> {run_dir / 'log.txt'}")
    return run_dir
