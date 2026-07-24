"""Thread-local logging context for structured metadata."""

from __future__ import annotations

import contextvars
import logging
from typing import Any

_log_context_var: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)


class LogContext:
    """Context manager that adds structured fields to all log records.

    Usage::

        with LogContext(experiment_id="abc-123", benchmark="agentboard"):
            logger.info("Benchmark starting")  # includes extra fields
    """

    def __init__(self, **kwargs: Any) -> None:
        self._fields = kwargs
        self._previous: dict[str, Any] = {}

    def __enter__(self) -> LogContext:
        token = _log_context_var.set({})
        ctx = _log_context_var.get()
        self._previous = dict(ctx)
        ctx.update(self._fields)
        return self

    def __exit__(self, *args: Any) -> None:
        _log_context_var.set({})


def get_log_context() -> dict[str, Any]:
    """Return the current thread-local logging context dict."""
    return dict(_log_context_var.get())
