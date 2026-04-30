import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)  # ← True (capital T)


def get_logger(name: str) -> logging.Logger:  # ← Logger (capital L)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # ← duplicate handlers se bachao

    logger.setLevel(logging.DEBUG)

    log_format = logging.Formatter(
        "[%(asctime)s] - [%(levelname)s] - [%(name)s] - %(message)s"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)  # ← logging.StreamHandler (not logger.)
    ch.setLevel(logging.INFO)
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    # File Handler
    fh = RotatingFileHandler(
        LOG_DIR / "pipeline.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(log_format)
    logger.addHandler(fh)

    return logger 