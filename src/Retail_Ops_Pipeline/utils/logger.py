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


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with:
    - Console handler (INFO level, colored output)
    - Rotating file handler (DEBUG level, max 10MB, 5 backups)

    Usage:
        logger = get_logger(__name__)
        logger.info("Pipeline started")
        logger.error("Something went wrong", exc_info=True)
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # ── Console Handler (INFO+) ───────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── File Handler (DEBUG+, rotating) ──────────────────────
    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
