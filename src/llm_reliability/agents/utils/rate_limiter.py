"""
Token-bucket rate limiter for LLM provider calls.

Enforces a maximum number of requests per second across repeated calls,
preventing providers from returning 429 errors due to burst traffic.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Thread-safe token-bucket rate limiter.

    Args:
        requests_per_second: Maximum allowed calls per second.

    Example::

        limiter = RateLimiter(requests_per_second=5)
        for task in tasks:
            limiter.acquire()
            response = provider.generate(task)
    """

    def __init__(self, requests_per_second: float = 10.0) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive.")
        self._interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._last_call: float = 0.0

    def acquire(self) -> None:
        """Block until the next permitted call slot is available."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            wait = self._interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
