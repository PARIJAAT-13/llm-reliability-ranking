"""Dynamic batching for inference runtimes."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class BatchStatistics:
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_duration_ms: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    batch_count: int = 0
    avg_batch_size: float = 0.0


@dataclass
class BatchResult:
    results: list[Any]
    errors: list[Exception] = field(default_factory=list)
    batch_size: int = 0
    batch_duration_ms: float = 0.0
    task_latencies_ms: list[float] = field(default_factory=list)


class TaskExecutor(Protocol):
    def execute(self, task: dict[str, Any]) -> Any: ...


class BatchProcessor:
    """Synchronous batch processor with configurable size and queue timeout."""

    def __init__(
        self,
        executor: TaskExecutor,
        max_batch_size: int = 16,
        queue_timeout: float = 0.1,
        auto_flush: bool = True,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if queue_timeout < 0:
            raise ValueError("queue_timeout must be >= 0")
        self._executor = executor
        self._max_batch_size = max_batch_size
        self._queue_timeout = queue_timeout
        self._auto_flush = auto_flush
        self._queue: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._stats = BatchStatistics()

    @property
    def stats(self) -> BatchStatistics:
        return self._stats

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    def submit(self, task: dict[str, Any]) -> None:
        with self._lock:
            self._queue.append(task)
            if self._auto_flush and len(self._queue) >= self._max_batch_size:
                self._flush_locked()

    def flush(self) -> BatchResult:
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> BatchResult:
        if not self._queue:
            return BatchResult(results=[])
        batch = self._queue[: self._max_batch_size]
        self._queue = self._queue[self._max_batch_size :]
        self._last_flush = time.monotonic()
        return self._execute_batch(batch)

    def flush_timeout(self) -> BatchResult | None:
        elapsed = time.monotonic() - self._last_flush
        if elapsed >= self._queue_timeout and self.queue_size > 0:
            return self.flush()
        return None

    def execute_batch(self, tasks: list[dict[str, Any]]) -> BatchResult:
        return self._execute_batch(tasks)

    def _execute_batch(self, tasks: list[dict[str, Any]]) -> BatchResult:
        results: list[Any] = []
        errors: list[Exception] = []
        latencies: list[float] = []
        t0 = time.perf_counter()
        for task in tasks:
            t1 = time.perf_counter()
            try:
                out = self._executor.execute(task)
                results.append(out)
                latencies.append((time.perf_counter() - t1) * 1000.0)
            except Exception as exc:
                errors.append(exc)
                results.append(None)
                latencies.append(0.0)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        completed_count = len([r for r in results if r is not None])
        self._update_stats(len(tasks), completed_count, len(errors), latencies, duration_ms)
        return BatchResult(
            results=results,
            errors=errors,
            batch_size=len(tasks),
            batch_duration_ms=duration_ms,
            task_latencies_ms=latencies,
        )

    def _update_stats(
        self,
        batch_size: int,
        completed: int,
        failed: int,
        latencies: list[float],
        duration_ms: float,
    ) -> None:
        s = self._stats
        s.total_tasks += batch_size
        s.completed_tasks += completed
        s.failed_tasks += failed
        s.total_duration_ms += duration_ms
        s.batch_count += 1
        s.avg_batch_size = s.total_tasks / s.batch_count
        valid = [ms for ms in latencies if ms > 0]
        if valid:
            s.avg_latency_ms = (
                s.avg_latency_ms * (s.batch_count - 1) + sum(valid) / len(valid)
            ) / s.batch_count
            if s.min_latency_ms == 0 or min(valid) < s.min_latency_ms:
                s.min_latency_ms = min(valid)
            if max(valid) > s.max_latency_ms:
                s.max_latency_ms = max(valid)

    def process_all(self, tasks: list[dict[str, Any]]) -> list[BatchResult]:
        batches: list[BatchResult] = []
        for i in range(0, len(tasks), self._max_batch_size):
            chunk = tasks[i : i + self._max_batch_size]
            batches.append(self.execute_batch(chunk))
        return batches


class AdaptiveBatcher:
    """Dynamically adjusts batch size based on observed latency."""

    def __init__(
        self,
        executor: TaskExecutor,
        initial_batch_size: int = 8,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
        target_latency_ms: float = 5000.0,
    ) -> None:
        self._executor = executor
        self._batch_size = initial_batch_size
        self._min = min_batch_size
        self._max = max_batch_size
        self._target = target_latency_ms
        self._history: list[float] = []

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def execute(self, tasks: list[dict[str, Any]]) -> list[Any]:
        results: list[Any] = []
        i = 0
        while i < len(tasks):
            current = self._batch_size
            chunk = tasks[i : i + current]
            t0 = time.perf_counter()
            for task in chunk:
                results.append(self._executor.execute(task))
            elapsed = (time.perf_counter() - t0) * 1000.0
            self._history.append(elapsed)
            self._adjust()
            i += current
        return results

    def _adjust(self) -> None:
        if not self._history:
            return
        recent = self._history[-5:]
        avg = sum(recent) / len(recent)
        if avg < self._target * 0.7 and self._batch_size < self._max:
            self._batch_size = min(self._batch_size * 2, self._max)
        elif avg > self._target * 1.3 and self._batch_size > self._min:
            self._batch_size = max(self._batch_size // 2, self._min)


# Backward-compatible alias
BatchExecutor = BatchProcessor
