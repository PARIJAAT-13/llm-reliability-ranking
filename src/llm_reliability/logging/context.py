"""Thread-local logging context for structured metadata."""

from __future__ import annotations

import contextvars
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
        self._token: contextvars.Token[dict[str, Any]] | None = None

    def __enter__(self) -> LogContext:
        token = _log_context_var.set({})
        orig = _log_context_var.get()
        self._previous = dict(orig)
        self._token = token
        orig.update(self._fields)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            _log_context_var.reset(self._token)


def get_log_context() -> dict[str, Any]:
    """Return the current thread-local logging context dict."""
    return dict(_log_context_var.get())
