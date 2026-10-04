"""Logging configuration."""
from __future__ import annotations

import logging
import sys
from typing import Optional

from core.config import get_settings


def setup_logging(level: Optional[str] = None) -> None:
    """Configure structured logging."""
    settings = get_settings()
    log_level = level or settings.log_level
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    
    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)