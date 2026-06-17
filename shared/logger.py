"""
Shared logging factory. All modules call get_logger(__name__) to obtain a logger
that writes JSON lines to logs/app.log and human-readable lines to stdout.

Never log raw PII — hash ID numbers with hash_id() before logging.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

_LOGS_DIR = Path(__file__).parent.parent / "logs"
_LOGS_DIR.mkdir(exist_ok=True)


# Attributes always present on a LogRecord — anything else was injected via the
# `extra={...}` argument and should be surfaced in the JSON payload.
_STD_LOGRECORD_ATTRS = frozenset(vars(logging.makeLogRecord({})).keys()) | {
    "message", "asctime", "taskName",
}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Merge any caller-supplied `extra={...}` fields (logging stores them as
        # top-level record attributes, not under record.extra).
        for key, value in record.__dict__.items():
            if key not in _STD_LOGRECORD_ATTRS and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


def _log_file(name: str) -> Path:
    if name.startswith("part1"):
        return _LOGS_DIR / "part1.log"
    if name.startswith("part2"):
        return _LOGS_DIR / "part2.log"
    return _LOGS_DIR / "app.log"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for *name*. Safe to call multiple times."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(_log_file(name), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JsonFormatter())
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(ch)

    return logger


def hash_id(id_number: str) -> str:
    """One-way SHA-256 hash of an ID number for safe logging (no raw PII)."""
    return hashlib.sha256(id_number.encode()).hexdigest()[:16]
