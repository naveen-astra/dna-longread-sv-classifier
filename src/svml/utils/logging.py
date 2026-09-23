"""Structured logging. Every module logs; every rule writes a log file (spec section 36)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)-28s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_configured = False


def setup_logging(logfile: str | Path | None = None, level: str = "INFO") -> None:
    """Configure root logging once, optionally teeing to a file."""
    global _configured

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if _configured:
        handlers_to_keep = [h for h in root.handlers if not isinstance(h, logging.FileHandler)]
        root.handlers = handlers_to_keep
    else:
        root.handlers = []
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        root.addHandler(stream)
        _configured = True

    if logfile is not None:
        path = Path(logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
