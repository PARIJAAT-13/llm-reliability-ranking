"""Log formatters — including JSON structured output."""

from __future__ import annotations

import json
import logging
from typing import Any

from llm_reliability.logging.context import get_log_context

# Standard LogRecord attribute names to exclude from extra fields.
_STANDARD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }
)


class ContextFilter(logging.Filter):
    """Attach thread-local ``LogContext`` fields to every log record.

    Install via ``addFilter`` on a logger or handler so that context fields
    are available to any formatter.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = get_log_context()
        for k, v in ctx.items():
            if k not in _STANDARD_ATTRS and not hasattr(record, k):
                setattr(record, k, v)
        return True


class JsonFormatter(logging.Formatter):
    """Format log records as newline-delimited JSON.

    Thread-local context fields from ``LogContext`` are included as top-level
    keys.  Custom attributes set via ``logging.LoggerAdapter`` or by passing
    ``extra`` to a log call are also included when they do not shadow standard
    LogRecord fields.

    Example output::

        {"time": "2026-07-25T00:42:00", "level": "INFO", "logger": "my.module",
         "message": "Benchmark starting", "experiment_id": "abc-123"}
    """

    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info and record.exc_info[0]:
            obj["exc_info"] = self.formatException(record.exc_info)

        # Include non-standard attributes set via extra or direct assignment
        for k, v in record.__dict__.items():
            if k not in _STANDARD_ATTRS and k not in obj:
                obj[k] = v

        # Thread-local context overrides extras with matching keys
        ctx = get_log_context()
        for k, v in ctx.items():
            if k not in _STANDARD_ATTRS:
                obj[k] = v

        return json.dumps(obj, default=str, ensure_ascii=False)
