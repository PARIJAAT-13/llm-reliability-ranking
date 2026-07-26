"""Comprehensive tests for runtime infrastructure (Phase 1).

Covers: batching, streaming, failover, cost accounting, hardware profiling.
Target: 150-300 meaningful tests.
"""

from __future__ import annotations

import threading
import time
import unittest.mock
from collections.abc import Generator
from decimal import Decimal
from typing import Any

import pytest

from llm_reliability.runtime.batching import (AdaptiveBatcher, BatchExecutor,
                                              BatchProcessor, BatchResult,
                                              BatchStatistics)
from llm_reliability.runtime.cost_accounting import (CostCalculator, CostEntry,
                                                     CostTracker, TokenAccount,
                                                     TokenUsage)
from llm_reliability.runtime.failover import (FailoverConfig, FailoverResult,
                                              FailoverStrategy,
                                              ProviderFailover, RetryConfig,
                                              RetryExecutor, RetryResult,
                                              RetryStrategy, compute_delay,
                                              is_retryable)
from llm_reliability.runtime.hardware_profiler import (CPUInfo, GPUInfo,
                                                       HardwareProfile,
                                                       HardwareProfiler,
                                                       MemoryInfo,
                                                       RuntimeStatistics,
                                                       RuntimeTimer)
from llm_reliability.runtime.streaming import (StreamAdapter,
                                               StreamingExecutor,
                                               StreamStatistics,
                                               TokenCollector, TokenStream,
                                               TokenStreamCollector)

# =========================================================================
# Mock helpers
# =========================================================================


class _MockExecutor:
    """A callable-compatible executor for TaskExecutor protocol."""

    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self._fail_ids = fail_ids or set()
        self.call_count = 0

    def execute(self, task: dict[str, Any]) -> str:
        self.call_count += 1
        tid = str(task.get("id", ""))
        if tid in self._fail_ids:
            msg = f"task {tid} failed"
            raise ValueError(msg)
        return f"result_{tid}"


class _StreamingMockExecutor:
    """An executor with stream_generate for streaming tests."""

    def execute(self, task: dict[str, Any]) -> str:
        return task.get("prompt", "")

    @staticmethod
    def stream_generate(task: dict[str, Any]) -> Generator[str, None, None]:
        text = task.get("prompt", "")
        yield from text


class _SlowExecutor:
    """An executor that introduces artificial delay."""

    def __init__(self, delay: float = 0.01) -> None:
        self._delay = delay

    def execute(self, task: dict[str, Any]) -> str:
        time.sleep(self._delay)
        return f"result_{task.get('id', '')}"


class _NonRetryableExecutor:
    """Raises non-retryable TypeError."""

    @staticmethod
    def execute(task: dict[str, Any]) -> Any:
        msg = "non-retryable error"
        raise TypeError(msg)


class _RuntimeProviderStub:
    """Minimal RuntimeProvider for failover tests."""

    def __init__(self, name: str, fail: bool = False, delay: float = 0.0) -> None:
        self._name = name
        self._fail = fail
        self._delay = delay
        self.initialized = False

    def initialize(self) -> None:
        self.initialized = True

    def execute(self, task: dict[str, Any]) -> str:
        if self._delay:
            time.sleep(self._delay)
        if self._fail:
            msg = f"{self._name} error"
            raise RuntimeError(msg)
        return f"{self._name}:{task.get('id', '')}"

    def shutdown(self) -> None:
        self.initialized = False


def _make_entry(**overrides: Any) -> CostEntry:
    """Helper to build a CostEntry with defaults."""
    defaults: dict[str, Any] = {
        "provider": "test",
        "model": "test-model",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": Decimal("0.01"),
        "latency_ms": 10.0,
    }
    defaults.update(overrides)
    return CostEntry(**defaults)


# =========================================================================
# Batching — BatchProcessor / BatchExecutor / BatchStatistics / BatchResult
# =========================================================================


class TestBatchProcessor:
    """BatchProcessor: submit, flush, execute_batch, process_all."""

    def test_create_default(self) -> None:
        bp = BatchProcessor(_MockExecutor())
        assert bp.queue_size == 0
        assert bp.stats.total_tasks == 0

    def test_create_invalid_max_batch_size_zero(self) -> None:
        with pytest.raises(ValueError, match="max_batch_size"):
            BatchProcessor(_MockExecutor(), max_batch_size=0)

    def test_create_invalid_max_batch_size_negative(self) -> None:
        with pytest.raises(ValueError, match="max_batch_size"):
            BatchProcessor(_MockExecutor(), max_batch_size=-1)

    def test_create_invalid_queue_timeout(self) -> None:
        with pytest.raises(ValueError, match="queue_timeout"):
            BatchProcessor(_MockExecutor(), queue_timeout=-0.1)

    def test_submit_single(self) -> None:
        bp = BatchProcessor(_MockExecutor(), max_batch_size=10, auto_flush=False)
        bp.submit({"id": "1"})
        assert bp.queue_size == 1

    def test_submit_auto_flush_triggers_at_capacity(self) -> None:
        bp = BatchProcessor(_MockExecutor(), max_batch_size=3, auto_flush=True)
        bp.submit({"id": "1"})
        bp.submit({"id": "2"})
        assert bp.queue_size == 2
        bp.submit({"id": "3"})
        assert bp.queue_size == 0

    def test_submit_auto_flush_disabled(self) -> None:
        bp = BatchProcessor(_MockExecutor(), max_batch_size=3, auto_flush=False)
        bp.submit({"id": "1"})
        bp.submit({"id": "2"})
        bp.submit({"id": "3"})
        assert bp.queue_size == 3

    def test_flush_returns_batch_results(self) -> None:
        bp = BatchProcessor(_MockExecutor(), auto_flush=False)
        bp.submit({"id": "a"})
        bp.submit({"id": "b"})
        result = bp.flush()
        assert isinstance(result, BatchResult)
        assert result.results == ["result_a", "result_b"]
        assert bp.queue_size == 0

    def test_flush_empty_queue(self) -> None:
        bp = BatchProcessor(_MockExecutor())
        result = bp.flush()
        assert isinstance(result, BatchResult)
        assert result.results == []

    def test_flush_timeout_triggers(self) -> None:
        bp = BatchProcessor(_MockExecutor(), queue_timeout=0.0, auto_flush=False)
        bp.submit({"id": "x"})
        result = bp.flush_timeout()
        assert isinstance(result, BatchResult)
        assert result.results == ["result_x"]

    def test_flush_timeout_not_elapsed(self) -> None:
        bp = BatchProcessor(_MockExecutor(), queue_timeout=10.0, auto_flush=False)
        bp.submit({"id": "x"})
        result = bp.flush_timeout()
        assert result is None

    def test_flush_timeout_empty_queue(self) -> None:
        bp = BatchProcessor(_MockExecutor(), queue_timeout=0.0)
        result = bp.flush_timeout()
        assert result is None

    def test_execute_batch_success(self) -> None:
        bp = BatchProcessor(_MockExecutor())
        tasks = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        result = bp.execute_batch(tasks)
        assert isinstance(result, BatchResult)
        assert result.results == ["result_1", "result_2", "result_3"]
        assert result.errors == []
        assert result.batch_size == 3

    def test_execute_batch_with_errors(self) -> None:
        bp = BatchProcessor(_MockExecutor(fail_ids={"bad"}))
        tasks = [{"id": "ok"}, {"id": "bad"}, {"id": "ok2"}]
        result = bp.execute_batch(tasks)
        assert result.results[0] == "result_ok"
        assert result.results[1] is None
        assert result.results[2] == "result_ok2"
        assert len(result.errors) == 1
        assert isinstance(result.errors[0], ValueError)

    def test_execute_batch_all_fail(self) -> None:
        bp = BatchProcessor(_MockExecutor(fail_ids={"1", "2"}))
        result = bp.execute_batch([{"id": "1"}, {"id": "2"}])
        assert len(result.errors) == 2
        assert len(result.results) == 2
        assert result.results[0] is None
        assert result.results[1] is None

    def test_execute_batch_empty(self) -> None:
        bp = BatchProcessor(_MockExecutor())
        result = bp.execute_batch([])
        assert result.results == []
        assert result.errors == []
        assert result.batch_size == 0

    def test_execute_batch_single_task(self) -> None:
        bp = BatchProcessor(_MockExecutor())
        result = bp.execute_batch([{"id": "only"}])
        assert result.results == ["result_only"]
        assert result.batch_size == 1

    def test_execute_batch_returns_latencies(self) -> None:
        bp = BatchProcessor(_MockExecutor())
        result = bp.execute_batch([{"id": "a"}])
        assert len(result.task_latencies_ms) == 1
        assert result.task_latencies_ms[0] >= 0

    def test_execute_batch_error_latency_is_zero(self) -> None:
        bp = BatchProcessor(_MockExecutor(fail_ids={"x"}))
        result = bp.execute_batch([{"id": "x"}])
        assert result.task_latencies_ms == [0.0]

    def test_process_all_single_batch(self) -> None:
        bp = BatchProcessor(_MockExecutor(), max_batch_size=10)
        tasks = [{"id": str(i)} for i in range(5)]
        batches = bp.process_all(tasks)
        assert len(batches) == 1
        assert batches[0].batch_size == 5

    def test_process_all_multiple_batches(self) -> None:
        bp = BatchProcessor(_MockExecutor(), max_batch_size=3)
        tasks = [{"id": str(i)} for i in range(10)]
        batches = bp.process_all(tasks)
        assert len(batches) == 4
        assert batches[0].batch_size == 3
        assert batches[1].batch_size == 3
        assert batches[2].batch_size == 3
        assert batches[3].batch_size == 1

    def test_process_all_empty(self) -> None:
        bp = BatchProcessor(_MockExecutor())
        batches = bp.process_all([])
        assert batches == []

    def test_stats_accumulate_on_execute_batch(self) -> None:
        bp = BatchProcessor(_MockExecutor())
        bp.execute_batch([{"id": "a"}])
        s = bp.stats
        assert s.total_tasks == 1
        assert s.completed_tasks == 1
        assert s.failed_tasks == 0
        assert s.batch_count == 1

    def test_stats_tracks_failures(self) -> None:
        bp = BatchProcessor(_MockExecutor(fail_ids={"x"}))
        bp.execute_batch([{"id": "x"}, {"id": "y"}])
        s = bp.stats
        assert s.total_tasks == 2
        assert s.completed_tasks == 1
        assert s.failed_tasks == 1

    def test_stats_updates_latencies(self) -> None:
        bp = BatchProcessor(_SlowExecutor(0.005))
        bp.execute_batch([{"id": "a"}])
        s = bp.stats
        assert s.avg_latency_ms > 0
        assert s.min_latency_ms > 0
        assert s.max_latency_ms > 0

    def test_stats_multiple_batches(self) -> None:
        bp = BatchProcessor(_MockExecutor(), max_batch_size=2)
        bp.execute_batch([{"id": "a"}, {"id": "b"}])
        bp.execute_batch([{"id": "c"}])
        s = bp.stats
        assert s.total_tasks == 3
        assert s.completed_tasks == 3
        assert s.batch_count == 2

    def test_concurrent_submit(self) -> None:
        bp = BatchProcessor(_MockExecutor(), max_batch_size=5, auto_flush=False)
        n = 20

        def submitter() -> None:
            for i in range(n):
                bp.submit({"id": str(i)})

        threads = [threading.Thread(target=submitter) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert bp.queue_size == 4 * n

    def test_executor_must_satisfy_protocol(self) -> None:
        class _Callable:
            def execute(self, task: dict[str, Any]) -> str:
                return "ok"

        bp = BatchProcessor(_Callable())
        result = bp.execute_batch([{"id": "1"}])
        assert result.results == ["ok"]

    def test_batch_result_type(self) -> None:
        bp = BatchProcessor(_MockExecutor())
        result = bp.execute_batch([{"id": "a"}])
        assert isinstance(result.batch_size, int)
        assert isinstance(result.batch_duration_ms, float)
        assert isinstance(result.task_latencies_ms, list)
        assert isinstance(result.errors, list)

    def test_batch_result_duration_positive(self) -> None:
        bp = BatchProcessor(_MockExecutor())
        result = bp.execute_batch([{"id": "a"}])
        assert result.batch_duration_ms >= 0.0

    @pytest.mark.parametrize("n", [0, 1, 5, 20])
    def test_execute_batch_various_sizes(self, n: int) -> None:
        bp = BatchProcessor(_MockExecutor())
        tasks = [{"id": str(i)} for i in range(n)]
        result = bp.execute_batch(tasks)
        assert result.batch_size == n
        assert len(result.results) == n

    def test_executor_error_preserves_other_results(self) -> None:
        bp = BatchProcessor(_MockExecutor(fail_ids={"2"}))
        result = bp.execute_batch([{"id": "1"}, {"id": "2"}, {"id": "3"}])
        assert result.results[0] == "result_1"
        assert result.results[2] == "result_3"

    def test_submit_idempotent_no_side_effects(self) -> None:
        executor = _MockExecutor()
        bp = BatchProcessor(executor, auto_flush=False)
        bp.submit({"id": "x"})
        assert executor.call_count == 0
        bp.flush()
        assert executor.call_count == 1


class TestBatchExecutorAlias:
    """BatchExecutor is a backward-compatible alias for BatchProcessor."""

    def test_alias(self) -> None:
        executor = BatchExecutor(_MockExecutor())
        assert isinstance(executor, BatchProcessor)

    def test_alias_execute_batch(self) -> None:
        executor = BatchExecutor(_MockExecutor())
        result = executor.execute_batch([{"id": "a"}])
        assert result.results == ["result_a"]


class TestBatchStatistics:
    """BatchStatistics dataclass."""

    def test_defaults(self) -> None:
        s = BatchStatistics()
        assert s.total_tasks == 0
        assert s.completed_tasks == 0
        assert s.failed_tasks == 0
        assert s.total_duration_ms == 0.0
        assert s.batch_count == 0

    def test_avg_batch_size(self) -> None:
        s = BatchStatistics(total_tasks=10, batch_count=2, avg_batch_size=5.0)
        assert s.avg_batch_size == 5.0

    def test_fields_mutable(self) -> None:
        s = BatchStatistics()
        s.total_tasks = 5
        assert s.total_tasks == 5


class TestBatchResult:
    """BatchResult dataclass."""

    def test_defaults(self) -> None:
        r = BatchResult(results=["a"])
        assert r.errors == []
        assert r.batch_size == 0
        assert r.task_latencies_ms == []

    def test_fields(self) -> None:
        r = BatchResult(
            results=["x", "y"],
            errors=[ValueError("e")],
            batch_size=2,
            batch_duration_ms=10.5,
            task_latencies_ms=[5.0, 0.0],
        )
        assert r.results == ["x", "y"]
        assert len(r.errors) == 1
        assert r.batch_size == 2
        assert r.batch_duration_ms == pytest.approx(10.5)


class TestAdaptiveBatcher:
    """AdaptiveBatcher: dynamic batch-size adjustment."""

    def test_default_initial_size(self) -> None:
        b = AdaptiveBatcher(_MockExecutor())
        assert b.batch_size == 8

    def test_custom_initial_size(self) -> None:
        b = AdaptiveBatcher(_MockExecutor(), initial_batch_size=16)
        assert b.batch_size == 16

    def test_execute_single_batch(self) -> None:
        b = AdaptiveBatcher(_MockExecutor())
        tasks = [{"id": "a"}, {"id": "b"}]
        results = b.execute(tasks)
        assert results == ["result_a", "result_b"]

    def test_execute_empty(self) -> None:
        b = AdaptiveBatcher(_MockExecutor())
        results = b.execute([])
        assert results == []

    def test_execute_large_task_list(self) -> None:
        b = AdaptiveBatcher(_MockExecutor(), initial_batch_size=5)
        tasks = [{"id": str(i)} for i in range(23)]
        results = b.execute(tasks)
        assert len(results) == 23

    def test_adjust_increase_when_fast(self) -> None:
        b = AdaptiveBatcher(
            _MockExecutor(), initial_batch_size=4, max_batch_size=64, target_latency_ms=5000.0
        )
        b._history = [100.0, 200.0, 150.0, 180.0, 120.0]
        old = b.batch_size
        b._adjust()
        assert b.batch_size >= old

    def test_adjust_decrease_when_slow(self) -> None:
        b = AdaptiveBatcher(
            _MockExecutor(), initial_batch_size=32, min_batch_size=1, target_latency_ms=500.0
        )
        b._history = [800.0, 900.0, 850.0, 950.0, 1000.0]
        old = b.batch_size
        b._adjust()
        assert b.batch_size < old

    def test_adjust_respects_max(self) -> None:
        b = AdaptiveBatcher(
            _MockExecutor(), initial_batch_size=32, max_batch_size=32, target_latency_ms=5000.0
        )
        b._history = [100.0] * 5
        b._adjust()
        assert b.batch_size == 32

    def test_adjust_respects_min(self) -> None:
        b = AdaptiveBatcher(
            _MockExecutor(), initial_batch_size=8, min_batch_size=8, target_latency_ms=100.0
        )
        b._history = [200.0] * 5
        b._adjust()
        assert b.batch_size == 8

    def test_adjust_no_history(self) -> None:
        b = AdaptiveBatcher(_MockExecutor(), initial_batch_size=16)
        b._adjust()
        assert b.batch_size == 16

    def test_adjust_in_range_does_nothing(self) -> None:
        b = AdaptiveBatcher(_MockExecutor(), initial_batch_size=16, target_latency_ms=1000.0)
        b._history = [700.0, 800.0, 750.0]
        old = b.batch_size
        b._adjust()
        assert b.batch_size == old

    def test_execute_records_history(self) -> None:
        b = AdaptiveBatcher(_MockExecutor())
        b.execute([{"id": "a"}])
        assert len(b._history) == 1

    def test_execute_with_errors(self) -> None:
        b = AdaptiveBatcher(_MockExecutor(fail_ids={"bad"}))
        tasks = [{"id": "ok"}, {"id": "bad"}]
        with pytest.raises(ValueError):
            b.execute(tasks)

    def test_batch_size_property(self) -> None:
        b = AdaptiveBatcher(_MockExecutor(), initial_batch_size=10)
        assert b.batch_size == 10


# =========================================================================
# Streaming — TokenStream / StreamingExecutor / TokenCollector / StreamStatistics
# =========================================================================


class TestTokenStream:
    """TokenStream: iteration, timeout, cancellation."""

    @staticmethod
    def _gen(*tokens: str) -> Generator[str, None, None]:
        yield from tokens

    def test_iterate_all_tokens(self) -> None:
        s = TokenStream(self._gen("a", "b", "c"))
        tokens = list(s)
        assert tokens == ["a", "b", "c"]

    def test_empty_stream(self) -> None:
        s = TokenStream(self._gen())
        tokens = list(s)
        assert tokens == []

    def test_none_generator(self) -> None:
        s = TokenStream()
        tokens = list(s)
        assert tokens == []

    def test_partial_output(self) -> None:
        s = TokenStream(self._gen("he", "llo"))
        for _ in s:
            pass
        assert s.partial_output == "hello"

    def test_token_count(self) -> None:
        s = TokenStream(self._gen("x", "y", "z"))
        for _ in s:
            pass
        assert s.token_count == 3

    def test_cancel_stops_iteration(self) -> None:
        gen = self._gen("a", "b", "c")
        s = TokenStream(gen)
        it = iter(s)
        assert next(it) == "a"
        s.cancel()
        remaining = list(it)
        assert remaining == []

    def test_cancelled_property(self) -> None:
        s = TokenStream(self._gen("a"))
        assert not s.cancelled
        s.cancel()
        assert s.cancelled

    def test_timeout_exceeded(self) -> None:
        def slow_gen() -> Generator[str, None, None]:
            time.sleep(0.5)
            yield "a"
            yield "b"

        s = TokenStream(slow_gen(), timeout=0.05)
        tokens = list(s)
        assert tokens == ["a"]

    def test_timed_out_property(self) -> None:
        def slow_gen() -> Generator[str, None, None]:
            yield "a"
            time.sleep(0.5)

        s = TokenStream(slow_gen(), timeout=0.05)
        for _ in s:
            pass
        assert s.timed_out

    def test_no_timeout(self) -> None:
        s = TokenStream(self._gen("a"), timeout=None)
        assert not s.timed_out

    def test_implements_iterable(self) -> None:
        s = TokenStream(self._gen("a", "b"))
        assert list(s) == ["a", "b"]

    def test_cancel_before_iteration(self) -> None:
        s = TokenStream(self._gen("a", "b"))
        s.cancel()
        tokens = list(s)
        assert tokens == []

    def test_multiple_iterations(self) -> None:
        s = TokenStream(self._gen("a"))
        first = list(s)
        second = list(s)
        assert first == ["a"]
        assert second == []


class TestStreamingExecutor:
    """StreamingExecutor: wraps executor for streaming."""

    def test_stream_default_generator_string(self) -> None:
        executor = _MockExecutor()
        se = StreamingExecutor(executor)
        stream = se.stream({"id": "test"})
        tokens = list(stream)
        assert tokens == ["result_test"]

    def test_stream_default_generator_non_string(self) -> None:
        class _IntExecutor:
            @staticmethod
            def execute(task: dict[str, Any]) -> int:
                return 42

        se = StreamingExecutor(_IntExecutor())
        stream = se.stream({})
        tokens = list(stream)
        assert tokens == ["42"]

    def test_stream_with_stream_generate(self) -> None:
        se = StreamingExecutor(_StreamingMockExecutor())
        stream = se.stream({"prompt": "ABC"})
        tokens = list(stream)
        assert tokens == ["A", "B", "C"]

    def test_stream_with_timeout(self) -> None:
        def delayed_gen(task: dict[str, Any]) -> Generator[str, None, None]:
            time.sleep(0.5)
            yield "a"
            yield "b"

        executor = _StreamingMockExecutor()
        executor.stream_generate = delayed_gen
        se = StreamingExecutor(executor)
        stream = se.stream({"prompt": "x"}, timeout=0.05)
        tokens = list(stream)
        assert tokens == ["a"]

    def test_stream_with_callbacks(self) -> None:
        executor = _MockExecutor()
        se = StreamingExecutor(executor)
        collected: list[str] = []
        se.add_callback(collected.append)
        results = list(se.stream_with_callbacks({"id": "test"}))
        assert results == ["result_test"]
        assert collected == ["result_test"]

    def test_stream_with_callbacks_multiple(self) -> None:
        executor = _MockExecutor()
        se = StreamingExecutor(executor)
        c1: list[str] = []
        c2: list[str] = []
        se.add_callback(c1.append)
        se.add_callback(c2.append)
        list(se.stream_with_callbacks({"id": "x"}))
        assert c1 == ["result_x"]
        assert c2 == ["result_x"]

    def test_stream_streaming_generator_with_callbacks(self) -> None:
        se = StreamingExecutor(_StreamingMockExecutor())
        collected: list[str] = []
        se.add_callback(collected.append)
        results = list(se.stream_with_callbacks({"prompt": "hi"}))
        assert results == ["h", "i"]
        assert collected == ["h", "i"]

    def test_stream_returns_token_stream(self) -> None:
        se = StreamingExecutor(_MockExecutor())
        stream = se.stream({"id": "a"})
        assert isinstance(stream, TokenStream)


class TestTokenCollector:
    """TokenCollector (and alias TokenStreamCollector)."""

    @staticmethod
    def _stream(*tokens: str) -> TokenStream:
        def gen() -> Generator[str, None, None]:
            yield from tokens

        return TokenStream(gen())

    def test_collect_full(self) -> None:
        result = TokenCollector.collect(self._stream("hello", " ", "world"))
        assert result == "hello world"

    def test_collect_empty(self) -> None:
        result = TokenCollector.collect(self._stream())
        assert result == ""

    def test_collect_single_token(self) -> None:
        result = TokenCollector.collect(self._stream("only"))
        assert result == "only"

    def test_collect_with_stats(self) -> None:
        text, stats = TokenCollector.collect_with_stats(self._stream("a", "b", "c"))
        assert text == "abc"
        assert isinstance(stats, StreamStatistics)
        assert stats.total_tokens == 3
        assert stats.total_duration_ms >= 0
        assert stats.tokens_per_second > 0

    def test_collect_with_stats_empty(self) -> None:
        text, stats = TokenCollector.collect_with_stats(self._stream())
        assert text == ""
        assert stats.total_tokens == 0
        assert stats.tokens_per_second == 0.0

    def test_collect_with_stats_cancelled(self) -> None:
        stream = self._stream("a", "b", "c")
        stream.cancel()
        text, stats = TokenCollector.collect_with_stats(stream)
        assert stats.cancelled

    def test_token_stream_collector_alias(self) -> None:
        assert TokenStreamCollector is TokenCollector

    def test_token_stream_collector_collect(self) -> None:
        result = TokenStreamCollector.collect(self._stream("x", "y"))
        assert result == "xy"


class TestStreamStatistics:
    """StreamStatistics dataclass."""

    def test_defaults(self) -> None:
        s = StreamStatistics()
        assert s.total_tokens == 0
        assert s.total_duration_ms == 0.0
        assert not s.cancelled
        assert not s.timed_out
        assert s.partial_output == ""


class TestStreamAdapterAlias:
    """StreamAdapter is backward-compatible alias for StreamingExecutor."""

    def test_alias(self) -> None:
        adapter = StreamAdapter(_MockExecutor())
        assert isinstance(adapter, StreamingExecutor)

    def test_stream(self) -> None:
        adapter = StreamAdapter(_MockExecutor())
        tokens = list(adapter.stream({"id": "a"}))
        assert tokens == ["result_a"]


# =========================================================================
# Failover — compute_delay / is_retryable / RetryExecutor / ProviderFailover
# =========================================================================


class TestComputeDelay:
    """compute_delay for various retry strategies."""

    def test_fixed(self) -> None:
        config = RetryConfig(strategy=RetryStrategy.FIXED, base_delay=2.0)
        assert compute_delay(config, 0) == pytest.approx(2.0)
        assert compute_delay(config, 3) == pytest.approx(2.0)

    def test_linear(self) -> None:
        config = RetryConfig(strategy=RetryStrategy.LINEAR, base_delay=1.0)
        assert compute_delay(config, 0) == pytest.approx(1.0)
        assert compute_delay(config, 1) == pytest.approx(2.0)
        assert compute_delay(config, 2) == pytest.approx(3.0)

    def test_exponential(self) -> None:
        config = RetryConfig(strategy=RetryStrategy.EXPONENTIAL, base_delay=1.0)
        assert compute_delay(config, 0) == pytest.approx(1.0)
        assert compute_delay(config, 1) == pytest.approx(2.0)
        assert compute_delay(config, 2) == pytest.approx(4.0)

    def test_jitter(self) -> None:
        config = RetryConfig(strategy=RetryStrategy.JITTER, base_delay=1.0)
        delays = {compute_delay(config, 2) for _ in range(20)}
        assert all(0.5 * 4 <= d <= 4.0 for d in delays)
        assert len(delays) > 1

    def test_max_delay_cap(self) -> None:
        config = RetryConfig(strategy=RetryStrategy.EXPONENTIAL, base_delay=100.0, max_delay=30.0)
        assert compute_delay(config, 10) == pytest.approx(30.0)

    def test_default_strategy_exponential(self) -> None:
        config = RetryConfig(base_delay=3.0)
        d0 = compute_delay(config, 0)
        d1 = compute_delay(config, 1)
        assert d1 == pytest.approx(d0 * 2)


class TestIsRetryable:
    """is_retryable checks if an exception should trigger retry."""

    def test_default_all_exceptions(self) -> None:
        config = RetryConfig()
        assert is_retryable(ValueError("x"), config)
        assert is_retryable(RuntimeError("x"), config)
        assert is_retryable(OSError("x"), config)

    def test_custom_retryable(self) -> None:
        config = RetryConfig(retryable_exceptions=(ValueError,))
        assert is_retryable(ValueError("x"), config)
        assert not is_retryable(RuntimeError("x"), config)
        assert not is_retryable(TypeError("x"), config)

    def test_subclass_match(self) -> None:
        class MyError(ValueError):
            pass

        config = RetryConfig(retryable_exceptions=(ValueError,))
        assert is_retryable(MyError("x"), config)

    def test_no_match(self) -> None:
        config = RetryConfig(retryable_exceptions=(TypeError,))
        assert not is_retryable(ValueError("x"), config)


class TestRetryExecutor:
    """RetryExecutor: retries a callable on failure."""

    def test_success_on_first_attempt(self) -> None:
        executor = _MockExecutor()
        re = RetryExecutor(executor.execute)
        result = re.execute({"id": "a"})
        assert result == "result_a"

    def test_retry_and_succeed(self) -> None:
        call_count = [0]

        def flaky(task: dict[str, Any]) -> str:
            call_count[0] += 1
            if call_count[0] < 3:
                msg = f"attempt {call_count[0]} failed"
                raise RuntimeError(msg)
            return "success"

        re = RetryExecutor(flaky, RetryConfig(max_attempts=5, base_delay=0.001))
        result = re.execute({})
        assert result == "success"
        assert call_count[0] == 3

    def test_exhaust_retries_raises(self) -> None:
        def always_fail(task: dict[str, Any]) -> Any:
            msg = "always fail"
            raise RuntimeError(msg)

        re = RetryExecutor(always_fail, RetryConfig(max_attempts=3, base_delay=0.001))
        with pytest.raises(RuntimeError, match="always fail"):
            re.execute({})

    def test_non_retryable_exception_immediate_raise(self) -> None:
        re = RetryExecutor(
            _NonRetryableExecutor().execute,
            RetryConfig(max_attempts=3, retryable_exceptions=(ValueError,), base_delay=0.001),
        )
        with pytest.raises(TypeError):
            re.execute({})

    def test_attempts_property_after_success(self) -> None:
        executor = _MockExecutor()
        re = RetryExecutor(executor.execute)
        re.execute({"id": "a"})
        assert len(re.attempts) == 1
        assert re.attempts[0].success
        assert re.attempts[0].attempt == 1

    def test_attempts_property_after_failure(self) -> None:
        def fail(task: dict[str, Any]) -> Any:
            msg = "fail"
            raise RuntimeError(msg)

        re = RetryExecutor(fail, RetryConfig(max_attempts=3, base_delay=0.001))
        with pytest.raises(RuntimeError):
            re.execute({})
        assert len(re.attempts) == 3
        assert all(not a.success for a in re.attempts)

    def test_retry_result_fields(self) -> None:
        executor = _MockExecutor()
        re = RetryExecutor(executor.execute)
        re.execute({"id": "a"})
        r = re.attempts[0]
        assert r.output == "result_a"
        assert r.attempt == 1
        assert r.total_delay >= 0
        assert r.success

    def test_retry_result_failure_fields(self) -> None:
        def fail(task: dict[str, Any]) -> Any:
            msg = "fail"
            raise RuntimeError(msg)

        re = RetryExecutor(fail, RetryConfig(max_attempts=1, base_delay=0.0))
        with pytest.raises(RuntimeError):
            re.execute({})
        r = re.attempts[0]
        assert r.output is None
        assert r.success is False

    def test_default_config(self) -> None:
        re = RetryExecutor(_MockExecutor().execute)
        assert re._config.max_attempts == 3
        assert re._config.strategy == RetryStrategy.EXPONENTIAL

    def test_attempts_cleared_between_calls(self) -> None:
        executor = _MockExecutor()
        re = RetryExecutor(executor.execute)
        re.execute({"id": "a"})
        assert len(re.attempts) == 1
        re.execute({"id": "b"})
        assert len(re.attempts) == 1

    def test_empty_task_dict(self) -> None:
        re = RetryExecutor(_MockExecutor().execute)
        result = re.execute({})
        assert result == "result_"

    def test_single_max_attempt(self) -> None:
        def fail(task: dict[str, Any]) -> Any:
            msg = "fail"
            raise RuntimeError(msg)

        re = RetryExecutor(fail, RetryConfig(max_attempts=1, base_delay=0.001))
        with pytest.raises(RuntimeError):
            re.execute({})
        assert len(re.attempts) == 1


class TestRetryConfig:
    """RetryConfig dataclass."""

    def test_defaults(self) -> None:
        c = RetryConfig()
        assert c.max_attempts == 3
        assert c.base_delay == 1.0
        assert c.max_delay == 60.0
        assert c.strategy == RetryStrategy.EXPONENTIAL

    def test_custom(self) -> None:
        c = RetryConfig(
            max_attempts=5, base_delay=0.5, max_delay=10.0, strategy=RetryStrategy.FIXED
        )
        assert c.max_attempts == 5
        assert c.base_delay == 0.5
        assert c.max_delay == 10.0
        assert c.strategy == RetryStrategy.FIXED


class TestProviderFailover:
    """ProviderFailover: failover across multiple providers."""

    def test_first_provider_succeeds(self) -> None:
        config = FailoverConfig(providers=["a", "b"], retry=RetryConfig(base_delay=0.001))
        fo = ProviderFailover(lambda name: _RuntimeProviderStub(name), config)
        result = fo.execute({"id": "t1"})
        assert result.provider == "a"
        assert result.output == "a:t1"
        assert result.attempt == 0
        assert result.success

    def test_failover_to_backup(self) -> None:
        config = FailoverConfig(providers=["a", "b"], retry=RetryConfig(base_delay=0.001))
        fo = ProviderFailover(
            lambda name: _RuntimeProviderStub(name, fail=(name == "a")),
            config,
        )
        result = fo.execute({"id": "t2"})
        assert result.provider == "b"
        assert result.output == "b:t2"
        assert result.attempt == 1

    def test_all_providers_fail(self) -> None:
        config = FailoverConfig(providers=["a", "b"], retry=RetryConfig(base_delay=0.001))
        fo = ProviderFailover(
            lambda name: _RuntimeProviderStub(name, fail=True),
            config,
        )
        with pytest.raises(RuntimeError, match="All 2 providers failed"):
            fo.execute({"id": "fail_all"})

    def test_single_provider(self) -> None:
        config = FailoverConfig(providers=["only"], retry=RetryConfig(base_delay=0.001))
        fo = ProviderFailover(lambda name: _RuntimeProviderStub(name), config)
        result = fo.execute({"id": "s"})
        assert result.provider == "only"

    def test_history_on_success(self) -> None:
        config = FailoverConfig(providers=["a"], retry=RetryConfig(base_delay=0.001))
        fo = ProviderFailover(lambda name: _RuntimeProviderStub(name), config)
        fo.execute({"id": "h1"})
        history = fo.history
        assert len(history) == 1
        assert history[0]["provider"] == "a"
        assert history[0]["success"] is True

    def test_history_on_failure(self) -> None:
        config = FailoverConfig(providers=["a", "b"], retry=RetryConfig(base_delay=0.001))
        fo = ProviderFailover(
            lambda name: _RuntimeProviderStub(name, fail=(name == "a")),
            config,
        )
        fo.execute({"id": "h2"})
        history = fo.history
        assert history[0]["provider"] == "a"
        assert history[0]["success"] is False
        assert "error" in history[0]

    def test_result_dataclass(self) -> None:
        r = FailoverResult(output="out", provider="p1", attempt=2, latency_ms=15.0)
        assert r.output == "out"
        assert r.provider == "p1"
        assert r.attempt == 2
        assert r.latency_ms == pytest.approx(15.0)
        assert r.success

    def test_failover_config_defaults(self) -> None:
        c = FailoverConfig(providers=["a", "b"])
        assert c.providers == ["a", "b"]
        assert c.strategy == FailoverStrategy.SEQUENTIAL
        assert c.retry.max_attempts == 3

    def test_failover_config_custom(self) -> None:
        c = FailoverConfig(
            providers=["x"],
            strategy=FailoverStrategy.ROUND_ROBIN,
            retry=RetryConfig(max_attempts=5, base_delay=0.5),
        )
        assert c.strategy == FailoverStrategy.ROUND_ROBIN
        assert c.retry.max_attempts == 5

    def test_round_robin_strategy(self) -> None:
        config = FailoverConfig(
            providers=["a", "b", "c"],
            strategy=FailoverStrategy.ROUND_ROBIN,
            retry=RetryConfig(base_delay=0.001),
        )
        fo = ProviderFailover(lambda name: _RuntimeProviderStub(name), config)
        result = fo.execute({"id": "rr"})
        assert result.provider in {"a", "b", "c"}

    def test_fastest_first_strategy(self) -> None:
        config = FailoverConfig(
            providers=["slow", "fast"],
            strategy=FailoverStrategy.FASTEST_FIRST,
            retry=RetryConfig(base_delay=0.001),
        )
        factories = {
            "slow": _RuntimeProviderStub("slow", delay=0.1),
            "fast": _RuntimeProviderStub("fast"),
        }

        def factory(name: str) -> _RuntimeProviderStub:
            return factories[name]

        fo = ProviderFailover(factory, config)
        fo._history = [
            {"provider": "fast", "success": True, "latency_ms": 1.0},
            {"provider": "slow", "success": True, "latency_ms": 100.0},
        ]
        result = fo.execute({"id": "ff"})
        assert result.provider == "fast"

    def test_provider_initialized_and_shutdown(self) -> None:
        config = FailoverConfig(providers=["a"], retry=RetryConfig(base_delay=0.001))
        stub = _RuntimeProviderStub("a")
        fo = ProviderFailover(lambda name: stub, config)
        fo.execute({"id": "init_test"})
        assert stub.initialized is False  # shutdown after execution

    def test_mixed_errors_and_success(self) -> None:
        config = FailoverConfig(providers=["p1", "p2", "p3"], retry=RetryConfig(base_delay=0.001))
        failures = {"p1", "p2"}

        fo = ProviderFailover(
            lambda name: _RuntimeProviderStub(name, fail=(name in failures)),
            config,
        )
        result = fo.execute({"id": "mix"})
        assert result.provider == "p3"
        assert result.attempt == 2

    def test_failover_with_empty_providers(self) -> None:
        config = FailoverConfig(providers=[], retry=RetryConfig(base_delay=0.001))
        fo = ProviderFailover(lambda name: _RuntimeProviderStub(name), config)
        with pytest.raises(RuntimeError, match="All 0 providers failed"):
            fo.execute({"id": "empty"})

    def test_history_property_returns_copy(self) -> None:
        config = FailoverConfig(providers=["a"], retry=RetryConfig(base_delay=0.001))
        fo = ProviderFailover(lambda name: _RuntimeProviderStub(name), config)
        fo.execute({"id": "cp"})
        h = fo.history
        h.append({"extra": True})
        assert len(fo.history) == 1

    def test_failover_result_latency_positive(self) -> None:
        config = FailoverConfig(providers=["a"], retry=RetryConfig(base_delay=0.001))
        fo = ProviderFailover(lambda name: _RuntimeProviderStub(name), config)
        result = fo.execute({"id": "lat"})
        assert result.latency_ms >= 0


# =========================================================================
# Cost accounting — CostCalculator / CostEntry / CostTracker / TokenUsage
# =========================================================================


class TestCostCalculator:
    """CostCalculator: estimate_cost, record_usage, register_pricing."""

    def test_estimate_openai_gpt4o(self) -> None:
        cost = CostCalculator.estimate_cost("openai", "gpt-4o", 1000, 500)
        expected = Decimal("2.50") + Decimal("5.00")
        assert cost == expected

    def test_estimate_anthropic_claude(self) -> None:
        cost = CostCalculator.estimate_cost("anthropic", "claude-3-5-sonnet-20241022", 2000, 1000)
        expected = Decimal("6.00") + Decimal("15.00")
        assert cost == expected

    def test_estimate_deepseek(self) -> None:
        cost = CostCalculator.estimate_cost("deepseek", "deepseek-chat", 1000, 1000)
        expected = Decimal("0.27") + Decimal("1.10")
        assert cost == expected

    def test_estimate_zero_tokens(self) -> None:
        cost = CostCalculator.estimate_cost("openai", "gpt-4o", 0, 0)
        assert cost == Decimal("0.0")

    def test_estimate_unknown_model(self) -> None:
        cost = CostCalculator.estimate_cost("openai", "nonexistent-v99", 1000, 500)
        assert cost == Decimal("0.0")

    def test_estimate_unknown_provider(self) -> None:
        cost = CostCalculator.estimate_cost("unknown_provider", "some-model", 100, 50)
        assert cost == Decimal("0.0")

    def test_estimate_pattern_matching(self) -> None:
        cost = CostCalculator.estimate_cost("mistral", "mistral-large-2407", 500, 200)
        expected = Decimal("1.00") + Decimal("1.20")
        assert cost == expected

    def test_estimate_google_gemini(self) -> None:
        cost = CostCalculator.estimate_cost("google", "gemini-2.5-pro-preview-03-25", 100, 50)
        expected = Decimal("0.125") + Decimal("0.25")
        assert cost == expected

    def test_register_pricing_new_provider(self) -> None:
        CostCalculator.register_pricing("custom_provider", "custom-model", 1.0, 2.0)
        cost = CostCalculator.estimate_cost("custom_provider", "custom-model", 1000, 500)
        expected = Decimal("1.00") + Decimal("1.00")
        assert cost == expected

    def test_register_pricing_updates_existing(self) -> None:
        CostCalculator.register_pricing("openai", "gpt-4o", 1.0, 2.0)
        cost = CostCalculator.estimate_cost("openai", "gpt-4o", 1000, 500)
        expected = Decimal("1.00") + Decimal("1.00")
        assert cost == expected
        CostCalculator._pricing["openai"]["gpt-4o"] = (2.5, 10.0)

    def test_estimate_cohere(self) -> None:
        cost = CostCalculator.estimate_cost("cohere", "command-r-plus", 1000, 500)
        expected = Decimal("2.50") + Decimal("5.00")
        assert cost == expected

    def test_estimate_mistral_small(self) -> None:
        cost = CostCalculator.estimate_cost("mistral", "mistral-small-2409", 100, 100)
        expected = Decimal("0.10") + Decimal("0.30")
        assert cost == expected

    def test_record_usage_creates_entry(self) -> None:
        entry = CostCalculator.record_usage("openai", "gpt-4o", 1000, 500, 120.5)
        assert isinstance(entry, CostEntry)
        assert entry.provider == "openai"
        assert entry.model == "gpt-4o"
        assert entry.input_tokens == 1000
        assert entry.output_tokens == 500
        assert entry.latency_ms == pytest.approx(120.5)
        assert entry.cost_usd > Decimal("0.0")

    def test_record_usage_with_request_duration(self) -> None:
        entry = CostCalculator.record_usage(
            "openai", "gpt-4o", 100, 50, 30.0, request_duration_ms=200.0
        )
        assert entry.request_duration_ms == pytest.approx(200.0)

    def test_record_usage_unknown_model_zero_cost(self) -> None:
        entry = CostCalculator.record_usage("unknown", "model", 100, 50, 10.0)
        assert entry.cost_usd == Decimal("0.0")

    def test_record_usage_timestamp(self) -> None:
        entry = CostCalculator.record_usage("openai", "gpt-4o", 10, 10, 5.0)
        assert isinstance(entry.timestamp, str)
        assert len(entry.timestamp) > 0


class TestCostEntry:
    """CostEntry dataclass."""

    def test_total_tokens_property(self) -> None:
        entry = CostEntry(
            provider="p",
            model="m",
            input_tokens=300,
            output_tokens=700,
            cost_usd=Decimal("1.0"),
            latency_ms=10.0,
        )
        assert entry.total_tokens == 1000

    def test_default_request_duration(self) -> None:
        entry = _make_entry()
        assert entry.request_duration_ms == 0.0

    def test_default_metadata(self) -> None:
        entry = _make_entry()
        assert entry.metadata == {}

    def test_custom_metadata(self) -> None:
        entry = _make_entry(metadata={"key": "val"})
        assert entry.metadata == {"key": "val"}

    def test_mutable_fields(self) -> None:
        entry = _make_entry()
        entry.input_tokens = 500
        assert entry.input_tokens == 500

    def test_timestamp_format(self) -> None:
        entry = _make_entry()
        assert "T" in entry.timestamp


class TestTokenUsage:
    """TokenUsage dataclass."""

    def test_total_property(self) -> None:
        usage = TokenUsage(input_tokens=300, output_tokens=700)
        assert usage.total == 1000

    def test_defaults(self) -> None:
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total == 0

    def test_zero_total(self) -> None:
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        assert usage.total == 0

    def test_large_values(self) -> None:
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=2_000_000)
        assert usage.total == 3_000_000


class TestCostTracker:
    """CostTracker (alias TokenAccount) tracks cumulative usage and cost."""

    def test_empty_initial(self) -> None:
        ct = CostTracker()
        assert ct.total_input_tokens == 0
        assert ct.total_output_tokens == 0
        assert ct.total_tokens == 0
        assert ct.total_cost_usd == Decimal("0.0")
        assert ct.entry_count == 0

    def test_add_entry(self) -> None:
        ct = CostTracker()
        entry = _make_entry(input_tokens=100, output_tokens=50, cost_usd=Decimal("0.01"))
        ct.add_entry(entry)
        assert ct.entry_count == 1
        assert ct.total_input_tokens == 100
        assert ct.total_output_tokens == 50
        assert ct.total_tokens == 150

    def test_add_entry_backward_compat(self) -> None:
        ct = CostTracker()
        entry = _make_entry()
        ct.add(entry)
        assert ct.entry_count == 1

    def test_record_call(self) -> None:
        ct = CostTracker()
        entry = ct.record_call("openai", "gpt-4o", 1000, 500, 50.0)
        assert isinstance(entry, CostEntry)
        assert ct.entry_count == 1
        assert ct.total_input_tokens == 1000

    def test_record_call_with_request_duration(self) -> None:
        ct = CostTracker()
        ct.record_call("openai", "gpt-4o", 100, 50, 30.0, request_duration_ms=150.0)
        assert ct.entries[0].request_duration_ms == pytest.approx(150.0)

    def test_multiple_entries(self) -> None:
        ct = CostTracker()
        ct.add_entry(_make_entry(input_tokens=100, output_tokens=50, cost_usd=Decimal("0.01")))
        ct.add_entry(_make_entry(input_tokens=200, output_tokens=100, cost_usd=Decimal("0.02")))
        assert ct.total_input_tokens == 300
        assert ct.total_output_tokens == 150
        assert ct.total_tokens == 450
        assert ct.entry_count == 2
        assert ct.total_cost_usd == Decimal("0.03")

    def test_total_cost_usd_sum(self) -> None:
        ct = CostTracker()
        ct.add_entry(_make_entry(cost_usd=Decimal("1.50")))
        ct.add_entry(_make_entry(cost_usd=Decimal("2.50")))
        assert ct.total_cost_usd == Decimal("4.00")

    def test_summary(self) -> None:
        ct = CostTracker()
        ct.add_entry(
            _make_entry(
                input_tokens=500, output_tokens=300, cost_usd=Decimal("0.50"), latency_ms=10.0
            )
        )
        summary = ct.summary()
        assert summary["total_input_tokens"] == 500
        assert summary["total_output_tokens"] == 300
        assert summary["total_tokens"] == 800
        assert summary["call_count"] == 1
        assert summary["total_cost_usd"] == 0.50
        assert summary["avg_latency_ms"] == 10.0

    def test_summary_empty(self) -> None:
        ct = CostTracker()
        summary = ct.summary()
        assert summary["total_tokens"] == 0
        assert summary["call_count"] == 0
        assert summary["total_cost_usd"] == 0.0
        assert summary["avg_latency_ms"] == 0.0

    def test_avg_latency(self) -> None:
        ct = CostTracker()
        ct.add_entry(_make_entry(latency_ms=10.0))
        ct.add_entry(_make_entry(latency_ms=20.0))
        assert ct.avg_latency_ms == pytest.approx(15.0)

    def test_avg_latency_empty(self) -> None:
        ct = CostTracker()
        assert ct.avg_latency_ms == 0.0

    def test_by_provider(self) -> None:
        ct = CostTracker()
        ct.add_entry(_make_entry(provider="p1"))
        ct.add_entry(_make_entry(provider="p2"))
        ct.add_entry(_make_entry(provider="p1"))
        grouped = ct.by_provider()
        assert len(grouped["p1"]) == 2
        assert len(grouped["p2"]) == 1

    def test_by_model(self) -> None:
        ct = CostTracker()
        ct.add_entry(_make_entry(model="m1"))
        ct.add_entry(_make_entry(model="m2"))
        ct.add_entry(_make_entry(model="m1"))
        grouped = ct.by_model()
        assert len(grouped["m1"]) == 2
        assert len(grouped["m2"]) == 1

    def test_entries_property_returns_copy(self) -> None:
        ct = CostTracker()
        ct.add_entry(_make_entry())
        entries = ct.entries
        entries.append(_make_entry())
        assert ct.entry_count == 1

    def test_token_account_alias(self) -> None:
        assert TokenAccount is CostTracker

    def test_token_account_tracks_totals(self) -> None:
        account: CostTracker = TokenAccount()
        entry1 = CostCalculator.record_usage("openai", "gpt-4o", 1000, 500, 50.0)
        entry2 = CostCalculator.record_usage("openai", "gpt-4o-mini", 2000, 1000, 30.0)
        account.add_entry(entry1)
        account.add_entry(entry2)
        assert account.total_input_tokens == 3000
        assert account.total_output_tokens == 1500
        assert account.total_tokens == 4500
        assert account.entry_count == 2

    def test_cost_tracker_entry_timestamps(self) -> None:
        ct = CostTracker()
        ct.record_call("openai", "gpt-4o", 10, 5, 1.0)
        assert isinstance(ct.entries[0].timestamp, str)


# =========================================================================
# Hardware profiling — HardwareProfiler / HardwareProfile / RuntimeTimer
# =========================================================================


class TestHardwareProfiler:
    """HardwareProfiler: profile, estimate_model_memory, can_run_model."""

    def test_profile_returns_hardware_profile(self) -> None:
        profile = HardwareProfiler.profile()
        assert isinstance(profile, HardwareProfile)
        assert isinstance(profile.platform, str)
        assert isinstance(profile.cpu.count, int)
        assert profile.cpu.count >= 1

    def test_profile_gpu_disabled(self) -> None:
        profile = HardwareProfiler.profile(gpu_enabled=False)
        assert not profile.gpu.available
        assert profile.gpu.count == 0

    def test_profile_has_cpu_info(self) -> None:
        profile = HardwareProfiler.profile()
        assert isinstance(profile.cpu, CPUInfo)
        assert isinstance(profile.cpu.count, int)

    def test_profile_has_memory_info(self) -> None:
        profile = HardwareProfiler.profile()
        assert isinstance(profile.memory, MemoryInfo)
        assert isinstance(profile.memory.total_gb, float)

    def test_profile_has_python_version(self) -> None:
        profile = HardwareProfiler.profile()
        assert isinstance(profile.python_version, str)
        assert len(profile.python_version) > 0

    def test_estimate_model_memory_fp32(self) -> None:
        mem = HardwareProfiler.estimate_model_memory(7.0)
        assert mem["fp32_gb"] == pytest.approx(28.0)

    def test_estimate_model_memory_fp16(self) -> None:
        mem = HardwareProfiler.estimate_model_memory(7.0)
        assert mem["fp16_gb"] == pytest.approx(14.0)

    def test_estimate_model_memory_int8(self) -> None:
        mem = HardwareProfiler.estimate_model_memory(7.0)
        assert mem["int8_gb"] == pytest.approx(7.0)

    def test_estimate_model_memory_int4(self) -> None:
        mem = HardwareProfiler.estimate_model_memory(7.0)
        assert mem["int4_gb"] == pytest.approx(3.5)

    def test_estimate_model_memory_zero_params(self) -> None:
        mem = HardwareProfiler.estimate_model_memory(0.0)
        assert mem["fp32_gb"] == 0.0
        assert mem["fp16_gb"] == 0.0
        assert mem["int8_gb"] == 0.0
        assert mem["int4_gb"] == 0.0

    def test_estimate_model_memory_returns_all_precisions(self) -> None:
        mem = HardwareProfiler.estimate_model_memory(1.0)
        assert set(mem.keys()) == {"fp32_gb", "fp16_gb", "int8_gb", "int4_gb"}

    def test_estimate_negative_params(self) -> None:
        mem = HardwareProfiler.estimate_model_memory(-1.0)
        assert mem["fp32_gb"] == pytest.approx(-4.0)

    def test_can_run_model_insufficient_memory(self) -> None:
        ok, msg = HardwareProfiler.can_run_model(1_000_000.0, precision="fp16", gpu_enabled=False)
        assert not ok
        assert "Insufficient" in msg

    def test_can_run_model_gpu_sufficient(self) -> None:
        profile = HardwareProfile(
            platform="Linux",
            cpu=CPUInfo(count=8),
            memory=MemoryInfo(total_gb=64.0, available_gb=32.0),
            gpu=GPUInfo(available=True, count=1, models=["A100"], vram_total_gb=[80.0]),
        )
        with unittest.mock.patch.object(HardwareProfiler, "profile", return_value=profile):
            ok, msg = HardwareProfiler.can_run_model(7.0, precision="fp16")
        assert ok
        assert "GPU" in msg

    def test_can_run_model_cpu_fallback(self) -> None:
        profile = HardwareProfile(
            platform="Linux",
            cpu=CPUInfo(count=16),
            memory=MemoryInfo(total_gb=128.0, available_gb=100.0),
            gpu=GPUInfo(available=True, count=1, models=["Small"], vram_total_gb=[4.0]),
        )
        with unittest.mock.patch.object(HardwareProfiler, "profile", return_value=profile):
            ok, msg = HardwareProfiler.can_run_model(7.0, precision="fp16")
        assert ok
        assert "CPU fallback" in msg

    def test_can_run_model_no_gpu_ram_fallback(self) -> None:
        profile = HardwareProfile(
            platform="Linux",
            cpu=CPUInfo(count=4),
            memory=MemoryInfo(total_gb=32.0, available_gb=16.0),
            gpu=GPUInfo(available=False),
        )
        with unittest.mock.patch.object(HardwareProfiler, "profile", return_value=profile):
            ok, msg = HardwareProfiler.can_run_model(3.0, precision="fp16")
        assert ok
        assert "CPU fallback" in msg

    def test_can_run_model_default_precision_fp16(self) -> None:
        ok, msg = HardwareProfiler.can_run_model(0.001, gpu_enabled=False)
        assert ok

    def test_can_run_model_no_gpu_cpu_large_enough(self) -> None:
        profile = HardwareProfile(
            platform="Linux",
            cpu=CPUInfo(count=4),
            memory=MemoryInfo(total_gb=32.0, available_gb=16.0),
            gpu=GPUInfo(available=False),
        )
        with unittest.mock.patch.object(HardwareProfiler, "profile", return_value=profile):
            ok, msg = HardwareProfiler.can_run_model(1.0, precision="fp16", gpu_enabled=False)
        assert ok


class TestHardwareProfile:
    """HardwareProfile and nested dataclasses."""

    def test_default_cpu(self) -> None:
        hp = HardwareProfile()
        assert hp.cpu.count == 0
        assert hp.cpu.model == ""

    def test_default_memory(self) -> None:
        hp = HardwareProfile()
        assert hp.memory.total_gb == 0.0

    def test_default_gpu(self) -> None:
        hp = HardwareProfile()
        assert not hp.gpu.available
        assert hp.gpu.count == 0
        assert hp.gpu.models == []
        assert hp.gpu.vram_total_gb == []

    def test_default_platform(self) -> None:
        hp = HardwareProfile()
        assert isinstance(hp.platform, str)

    def test_custom_gpu(self) -> None:
        gpu = GPUInfo(
            available=True,
            count=2,
            models=["A100", "V100"],
            vram_total_gb=[80.0, 32.0],
            metal_supported=False,
        )
        hp = HardwareProfile(gpu=gpu)
        assert hp.gpu.available
        assert hp.gpu.count == 2
        assert hp.gpu.models == ["A100", "V100"]

    def test_custom_cpu(self) -> None:
        cpu = CPUInfo(count=16, model="AMD EPYC")
        hp = HardwareProfile(cpu=cpu)
        assert hp.cpu.count == 16
        assert hp.cpu.model == "AMD EPYC"

    def test_custom_memory(self) -> None:
        mem = MemoryInfo(total_gb=128.0, available_gb=64.0)
        hp = HardwareProfile(memory=mem)
        assert hp.memory.total_gb == 128.0
        assert hp.memory.available_gb == 64.0

    def test_platform_set_explicitly(self) -> None:
        hp = HardwareProfile(platform="CustomOS")
        assert hp.platform == "CustomOS"

    def test_hostname_present(self) -> None:
        hp = HardwareProfile()
        assert isinstance(hp.hostname, str)

    def test_gpu_metal_supported(self) -> None:
        gpu = GPUInfo(metal_supported=True)
        hp = HardwareProfile(gpu=gpu)
        assert hp.gpu.metal_supported

    def test_gpu_cuda_version(self) -> None:
        gpu = GPUInfo(cuda_version="12.4")
        hp = HardwareProfile(gpu=gpu)
        assert hp.gpu.cuda_version == "12.4"

    def test_gpu_rocm_version(self) -> None:
        gpu = GPUInfo(rocm_version="6.0")
        hp = HardwareProfile(gpu=gpu)
        assert hp.gpu.rocm_version == "6.0"


class TestRuntimeTimer:
    """RuntimeTimer: measure execution time and accumulate statistics."""

    def test_measure_returns_result(self) -> None:
        timer = RuntimeTimer()
        result = timer.measure(lambda: 42)
        assert result == 42

    def test_measure_records_stats(self) -> None:
        timer = RuntimeTimer()
        timer.measure(lambda: None)
        assert timer.stats.execution_count == 1
        assert timer.stats.total_time_ms > 0
        assert timer.stats.avg_time_ms > 0

    def test_measure_multiple_calls(self) -> None:
        timer = RuntimeTimer()
        timer.measure(lambda: None)
        timer.measure(lambda: None)
        timer.measure(lambda: None)
        assert timer.stats.execution_count == 3

    def test_measure_tracks_min_max(self) -> None:
        timer = RuntimeTimer()
        timer.measure(lambda: time.sleep(0.01))
        timer.measure(lambda: None)
        assert timer.stats.min_time_ms < timer.stats.max_time_ms

    def test_measure_iteration(self) -> None:
        timer = RuntimeTimer()
        latencies = timer.measure_iteration(lambda: None, iterations=5)
        assert len(latencies) == 5
        assert timer.stats.execution_count == 5
        assert timer.stats.avg_time_ms > 0

    def test_measure_iteration_single(self) -> None:
        timer = RuntimeTimer()
        latencies = timer.measure_iteration(lambda: 99, iterations=1)
        assert len(latencies) == 1
        assert timer.stats.execution_count == 1

    def test_reset_clears_stats(self) -> None:
        timer = RuntimeTimer()
        timer.measure(lambda: None)
        assert timer.stats.execution_count == 1
        timer.reset()
        assert timer.stats.execution_count == 0
        assert timer.stats.total_time_ms == 0.0
        assert timer.stats.avg_time_ms == 0.0

    def test_measure_error_still_records(self) -> None:
        timer = RuntimeTimer()
        with pytest.raises(ValueError):
            timer.measure(lambda: (_ for _ in ()).throw(ValueError("bad")))
        assert timer.stats.execution_count == 1
        assert timer.stats.total_time_ms > 0

    def test_stats_initial_defaults(self) -> None:
        timer = RuntimeTimer()
        s = timer.stats
        assert s.execution_count == 0
        assert s.total_time_ms == 0.0
        assert s.avg_time_ms == 0.0
        assert s.min_time_ms == 0.0
        assert s.max_time_ms == 0.0

    def test_stats_avg_calculation(self) -> None:
        timer = RuntimeTimer()
        timer.measure(lambda: None)
        timer.measure(lambda: None)
        assert timer.stats.avg_time_ms == pytest.approx(timer.stats.total_time_ms / 2)


class TestRuntimeStatistics:
    """RuntimeStatistics dataclass."""

    def test_defaults(self) -> None:
        s = RuntimeStatistics()
        assert s.execution_count == 0
        assert s.total_time_ms == 0.0
        assert s.avg_time_ms == 0.0
        assert s.min_time_ms == 0.0
        assert s.max_time_ms == 0.0

    def test_fields_mutable(self) -> None:
        s = RuntimeStatistics(execution_count=5, total_time_ms=100.0)
        s.avg_time_ms = 20.0
        assert s.avg_time_ms == 20.0


class TestCPUInfo:
    """CPUInfo dataclass."""

    def test_defaults(self) -> None:
        c = CPUInfo()
        assert c.count == 0
        assert c.model == ""
        assert c.usage_percent == 0.0

    def test_custom(self) -> None:
        c = CPUInfo(count=8, model="Intel", usage_percent=45.0)
        assert c.count == 8
        assert c.model == "Intel"
        assert c.usage_percent == 45.0


class TestMemoryInfo:
    """MemoryInfo dataclass."""

    def test_defaults(self) -> None:
        m = MemoryInfo()
        assert m.total_gb == 0.0
        assert m.available_gb == 0.0
        assert m.used_gb == 0.0
        assert m.percent == 0.0

    def test_custom(self) -> None:
        m = MemoryInfo(total_gb=32.0, available_gb=16.0, used_gb=16.0, percent=50.0)
        assert m.total_gb == 32.0
        assert m.available_gb == 16.0
        assert m.percent == 50.0


class TestGPUInfo:
    """GPUInfo dataclass."""

    def test_defaults(self) -> None:
        g = GPUInfo()
        assert not g.available
        assert g.count == 0
        assert g.models == []
        assert g.vram_total_gb == []
        assert g.vram_free_gb == []
        assert g.cuda_version is None
        assert g.rocm_version is None
        assert not g.metal_supported

    def test_custom(self) -> None:
        g = GPUInfo(
            available=True, count=1, models=["RTX 4090"], vram_total_gb=[24.0], cuda_version="12.4"
        )
        assert g.available
        assert g.count == 1
        assert g.models == ["RTX 4090"]
        assert g.cuda_version == "12.4"


# =========================================================================
# Integration tests — cross-module scenarios
# =========================================================================


class TestBatchAndStreamIntegration:
    """Combined scenarios using multiple runtime modules."""

    def test_batch_processor_with_retry(self) -> None:
        call_count = [0]

        def flaky(task: dict[str, Any]) -> str:
            call_count[0] += 1
            if call_count[0] < 3:
                msg = f"fail {call_count[0]}"
                raise RuntimeError(msg)
            return "ok"

        retry = RetryExecutor(flaky, RetryConfig(max_attempts=5, base_delay=0.001))
        bp = BatchProcessor(retry)
        result = bp.execute_batch([{"id": "1"}, {"id": "2"}])
        assert result.results == ["ok", "ok"]
        assert call_count[0] == 4

    def test_stream_with_cost_tracking(self) -> None:
        executor = _MockExecutor()
        se = StreamingExecutor(executor)
        stream = se.stream({"id": "test"})
        text = TokenCollector.collect(stream)
        ct = CostTracker()
        ct.record_call("test", "stream-model", len(text), 0, 10.0)
        assert ct.total_tokens == 11

    def test_failover_retry_config_used(self) -> None:
        config = FailoverConfig(
            providers=["a", "b"],
            retry=RetryConfig(max_attempts=2, base_delay=0.001),
        )
        fo = ProviderFailover(
            lambda name: _RuntimeProviderStub(name, fail=True),
            config,
        )
        with pytest.raises(RuntimeError, match="All 2 providers failed"):
            fo.execute({"id": "fail"})
        assert len(fo.history) == 2

    def test_hardware_profile_with_cost_estimation(self) -> None:
        profile = HardwareProfiler.profile(gpu_enabled=False)
        mem = HardwareProfiler.estimate_model_memory(7.0)
        cost = CostCalculator.estimate_cost("openai", "gpt-4o", 1000, 500)
        assert profile.cpu.count >= 1
        assert mem["fp16_gb"] == 14.0
        assert cost > Decimal("0.0")

    def test_adaptive_batcher_with_retry_executor(self) -> None:
        retry = RetryExecutor(
            _MockExecutor().execute, RetryConfig(max_attempts=2, base_delay=0.001)
        )
        ab = AdaptiveBatcher(retry, initial_batch_size=4)
        tasks = [{"id": str(i)} for i in range(10)]
        results = ab.execute(tasks)
        assert len(results) == 10

    def test_batch_result_feed_into_cost_tracker(self) -> None:
        executor = _MockExecutor()
        bp = BatchProcessor(executor)
        result = bp.execute_batch([{"id": "a"}, {"id": "b"}])
        ct = CostTracker()
        for r in result.results:
            ct.record_call("test", "batch-model", len(r), 0, 5.0)
        assert ct.entry_count == 2

    def test_runtime_timer_measures_batch(self) -> None:
        timer = RuntimeTimer()
        bp = BatchProcessor(_MockExecutor())
        result = timer.measure(lambda: bp.execute_batch([{"id": "a"}]))
        assert isinstance(result, BatchResult)
        assert timer.stats.execution_count == 1

    def test_token_stream_with_cost_tracker(self) -> None:
        def gen() -> Generator[str, None, None]:
            yield from "hello"

        stream = TokenStream(gen())
        list(stream)
        ct = CostTracker()
        ct.record_call("test", "stream-model", stream.token_count, 0, 2.0)
        assert ct.total_input_tokens == 5

    def test_provider_failover_with_timer(self) -> None:
        timer = RuntimeTimer()
        config = FailoverConfig(providers=["a"], retry=RetryConfig(base_delay=0.001))
        fo = ProviderFailover(lambda name: _RuntimeProviderStub(name), config)
        result = timer.measure(lambda: fo.execute({"id": "t"}))
        assert isinstance(result, FailoverResult)
        assert timer.stats.execution_count == 1
