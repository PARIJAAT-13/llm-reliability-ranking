"""
Retry utilities for LLM provider calls.

Provides a standalone retry decorator and a functional helper so
adapters can apply retry logic declaratively.
"""

import functools
import logging
import time
from typing import Callable, TypeVar

from llm_reliability.agents.adapters.exceptions import ProviderError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def with_retry(
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
    retryable: type[Exception] = ProviderError,
) -> Callable[[F], F]:
    """Decorator that retries a callable on retryable exceptions.

    Args:
        max_attempts:    Maximum number of attempts (including the first).
        backoff_seconds: Base wait time; doubles after each failure.
        retryable:       Exception class (or base class) that triggers a retry.

    Returns:
        A decorator that wraps the target function with retry logic.
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable as exc:
                    last_exc = exc
                    wait = backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Retryable error on attempt %d/%d: %s — retrying in %.1fs",
                        attempt, max_attempts, exc, wait,
                    )
                    if attempt < max_attempts:
                        time.sleep(wait)
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator
