"""Lightweight performance profiling for experiment runs.

Tracks total runtime, per-benchmark and per-model durations, and
cache effectiveness.  No external profilers required.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Lightweight in-process profiler for experiment execution.

    Usage::

        profiler = PerformanceProfiler()
        profiler.start_experiment()

        with profiler.measure_benchmark("agentboard", "gpt-4"):
            ...

        report = profiler.summary()
    """

    def __init__(self) -> None:
        self._experiment_start: float | None = None
        self._benchmark_times: dict[tuple[str, str], float] = defaultdict(float)
        self._benchmark_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._num_cache_hits: int = 0
        self._num_cache_misses: int = 0
        self._num_errors: int = 0

    def start_experiment(self) -> None:
        """Mark the start of an experiment."""
        self._experiment_start = time.perf_counter()

    @property
    def experiment_elapsed(self) -> float:
        """Return seconds since ``start_experiment()``, or 0."""
        if self._experiment_start is None:
            return 0.0
        return time.perf_counter() - self._experiment_start

    def record_cache_hit(self) -> None:
        self._num_cache_hits += 1

    def record_cache_miss(self) -> None:
        self._num_cache_misses += 1

    def record_error(self) -> None:
        self._num_errors += 1

    def measure_benchmark(self, benchmark: str, model: str) -> _BenchmarkTimer:
        """Return a context manager that times a benchmark run.

        Usage::

            with profiler.measure_benchmark("agentboard", "gpt-4"):
                ...  # benchmark execution
        """
        return _BenchmarkTimer(self, benchmark, model)

    def summary(self) -> dict[str, Any]:
        """Return a snapshot of collected profiling data."""
        cache_total = self._num_cache_hits + self._num_cache_misses
        cache_rate = (self._num_cache_hits / cache_total * 100) if cache_total > 0 else None

        per_benchmark: dict[str, Any] = {}
        for (benchmark, model), total_time in sorted(self._benchmark_times.items()):
            key = f"{benchmark}/{model}"
            count = self._benchmark_counts.get((benchmark, model), 0)
            per_benchmark[key] = {
                "benchmark": benchmark,
                "model": model,
                "total_seconds": round(total_time, 3),
                "run_count": count,
                "avg_seconds": round(total_time / count, 3) if count > 0 else 0,
            }

        return {
            "total_seconds": round(self.experiment_elapsed, 3),
            "per_benchmark": per_benchmark,
            "cache_hits": self._num_cache_hits,
            "cache_misses": self._num_cache_misses,
            "cache_hit_rate_pct": (round(cache_rate, 1) if cache_rate is not None else None),
            "errors": self._num_errors,
        }


class _BenchmarkTimer:
    """Context manager returned by ``PerformanceProfiler.measure_benchmark()``."""

    def __init__(self, profiler: PerformanceProfiler, benchmark: str, model: str) -> None:
        self._profiler = profiler
        self._benchmark = benchmark
        self._model = model
        self._start: float | None = None

    def __enter__(self) -> _BenchmarkTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = time.perf_counter() - (self._start or 0)
        key = (self._benchmark, self._model)
        self._profiler._benchmark_times[key] += elapsed
        self._profiler._benchmark_counts[key] += 1
