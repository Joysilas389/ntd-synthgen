"""Structured logging configuration for the NTD synthetic data engine."""
from __future__ import annotations

import logging
import sys
from logging import Logger


_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "ntd_synthgen") -> Logger:
    """Return a configured logger.

    Idempotent: calling twice with the same name will not duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
