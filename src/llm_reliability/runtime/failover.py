"""Retry and failover for inference runtimes."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class RetryStrategy(str, Enum):
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    JITTER = "jitter"


class FailoverStrategy(str, Enum):
    SEQUENTIAL = "sequential"
    ROUND_ROBIN = "round_robin"
    FASTEST_FIRST = "fastest_first"


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)


@dataclass
class FailoverConfig:
    providers: list[str]
    strategy: FailoverStrategy = FailoverStrategy.SEQUENTIAL
    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass
class RetryResult:
    output: Any
    attempt: int
    total_delay: float
    success: bool = True


@dataclass
class FailoverResult:
    output: Any
    provider: str
    attempt: int
    latency_ms: float
    success: bool = True


class RuntimeProvider(Protocol):
    def initialize(self) -> None: ...
    def execute(self, task: dict[str, Any]) -> Any: ...
    def shutdown(self) -> None: ...


def compute_delay(config: RetryConfig, attempt: int) -> float:
    if config.strategy == RetryStrategy.FIXED:
        delay = config.base_delay
    elif config.strategy == RetryStrategy.LINEAR:
        delay = config.base_delay * (attempt + 1)
    elif config.strategy == RetryStrategy.JITTER:
        delay = config.base_delay * (2**attempt)
        delay = delay * (0.5 + random.random() * 0.5)
    else:
        delay = config.base_delay * (2**attempt)
    return min(delay, config.max_delay)


def is_retryable(exception: Exception, config: RetryConfig) -> bool:
    return isinstance(exception, config.retryable_exceptions)


class RetryExecutor:
    """Wraps a callable with configurable retry logic."""

    def __init__(
        self,
        execute_fn: Callable[[dict[str, Any]], Any],
        config: RetryConfig | None = None,
    ) -> None:
        self._execute_fn = execute_fn
        self._config = config or RetryConfig()
        self._attempts: list[RetryResult] = []

    @property
    def attempts(self) -> list[RetryResult]:
        return list(self._attempts)

    def execute(self, task: dict[str, Any]) -> Any:
        self._attempts.clear()
        total_delay = 0.0
        last_exc: Exception | None = None
        for attempt in range(self._config.max_attempts):
            try:
                output = self._execute_fn(task)
                self._attempts.append(
                    RetryResult(output=output, attempt=attempt + 1, total_delay=total_delay)
                )
                return output
            except Exception as exc:
                last_exc = exc
                self._attempts.append(
                    RetryResult(
                        output=None,
                        attempt=attempt + 1,
                        total_delay=total_delay,
                        success=False,
                    )
                )
                if not is_retryable(exc, self._config) or attempt == self._config.max_attempts - 1:
                    raise
                delay = compute_delay(self._config, attempt)
                total_delay += delay
                logger.warning(
                    "Retryable error on attempt %d/%d: %s. Retrying in %.2fs",
                    attempt + 1,
                    self._config.max_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
        if last_exc is not None:
            raise last_exc
        msg = "All retry attempts exhausted"
        raise RuntimeError(msg)


class ProviderFailover:
    """Executes a task across multiple providers with failover."""

    def __init__(
        self,
        runtime_factory: Callable[[str], RuntimeProvider],
        config: FailoverConfig,
    ) -> None:
        self._factory = runtime_factory
        self._config = config
        self._history: list[dict[str, Any]] = []

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def execute(self, task: dict[str, Any]) -> FailoverResult:
        providers = self._resolve_order()
        for attempt, provider_name in enumerate(providers):
            runtime = self._factory(provider_name)
            try:
                runtime.initialize()
                t0 = time.perf_counter()
                output = runtime.execute(task)
                latency = (time.perf_counter() - t0) * 1000.0
                self._history.append(
                    {
                        "provider": provider_name,
                        "success": True,
                        "latency_ms": latency,
                    }
                )
                return FailoverResult(
                    output=output,
                    provider=provider_name,
                    attempt=attempt,
                    latency_ms=latency,
                )
            except Exception as exc:
                logger.warning(
                    "Provider %s failed (attempt %d/%d): %s",
                    provider_name,
                    attempt + 1,
                    len(providers),
                    exc,
                )
                self._history.append(
                    {
                        "provider": provider_name,
                        "success": False,
                        "error": str(exc),
                    }
                )
                delay = compute_delay(self._config.retry, attempt)
                time.sleep(delay)
            finally:
                runtime.shutdown()
        raise RuntimeError(f"All {len(providers)} providers failed for task.")

    def _resolve_order(self) -> list[str]:
        if self._config.strategy == FailoverStrategy.ROUND_ROBIN:
            shuffled = list(self._config.providers)
            random.shuffle(shuffled)
            return shuffled
        elif self._config.strategy == FailoverStrategy.FASTEST_FIRST:
            return sorted(self._config.providers, key=lambda p: self._avg_latency(p))
        return list(self._config.providers)

    def _avg_latency(self, provider: str) -> float:
        entries = [h for h in self._history if h["provider"] == provider and h.get("success")]
        if not entries:
            return float("inf")
        return sum(e["latency_ms"] for e in entries) / len(entries)
