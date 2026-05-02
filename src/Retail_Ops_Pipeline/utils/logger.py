"""
Logger — Structured, file + console logging for the entire pipeline.
Import this in every module: from Retail_Ops_Pipeline.utils.logger import get_logger
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from Retail_Ops_Pipeline.constants.config_entity import (
    LOG_DIR,
    LOG_FILE,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
)

# Create log directory if it doesn't exist
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, to_console: bool = False) -> logging.Logger:
    """
    Returns a logger with:
    - Console handler (Optional, only if to_console=True)
    - Rotating file handler (DEBUG level, always active)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # ── File Handler (Always DEBUG+, rotating) ──────────────────────
    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # ── Console Handler (Only if explicitly requested) ─────────────
    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
