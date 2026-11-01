"""
utils/logger.py
===============
Configures a project-wide Python logger that writes to both stdout
and an optional log file.

Usage
-----
    from src.utils.logger import get_logger

    log = get_logger(__name__)
    log.info("Training started.")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


# Track handler-attached loggers to avoid duplicate handlers on re-import.
_CONFIGURED_LOGGERS: set[str] = set()


def get_logger(
    name: str = "pet_seg",
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Return a named logger with console (and optional file) handlers.

    Parameters
    ----------
    name:
        Logger name — typically ``__name__`` of the calling module.
    level:
        Logging level string: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``.  Defaults to ``"INFO"``.
    log_file:
        Optional path to a ``.log`` file.  Parent directories are
        created automatically.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Skip reconfiguration on repeated imports.
    if name in _CONFIGURED_LOGGERS:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ── File handler (optional) ───────────────────────────────────────────────
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    # Prevent propagation to the root logger (avoids duplicate messages).
    logger.propagate = False

    _CONFIGURED_LOGGERS.add(name)
    return logger


def configure_root_logger(
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> None:
    """
    Configure the root logger.  Call once at application entry-point.

    Parameters
    ----------
    level:
        Logging verbosity.
    log_file:
        Optional path to a log file.
    """
    get_logger("pet_seg", level=level, log_file=log_file)

