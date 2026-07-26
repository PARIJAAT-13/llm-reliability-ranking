"""Streaming support for inference runtimes."""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class StreamStatistics:
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    tokens_per_second: float = 0.0
    cancelled: bool = False
    timed_out: bool = False
    partial_output: str = ""


class StreamCallback(Protocol):
    def __call__(self, token: str) -> None: ...


class TokenStream:
    """An iterable stream of tokens with cancellation and timeout support."""

    def __init__(
        self,
        generator: Generator[str, None, None] | None = None,
        timeout: float | None = None,
    ) -> None:
        self._generator = generator
        self._timeout = timeout
        self._cancelled = False
        self._start_time: float | None = None
        self._tokens: list[str] = []

    def __iter__(self) -> TokenStream:
        return self

    def __next__(self) -> str:
        if self._cancelled:
            raise StopIteration()
        if self._generator is None:
            raise StopIteration()
        if self._start_time is None:
            self._start_time = time.monotonic()
        if self._timeout is not None:
            elapsed = time.monotonic() - self._start_time
            if elapsed >= self._timeout:
                self._cancelled = True
                raise StopIteration()
        try:
            token = next(self._generator)
            self._tokens.append(token)
            return token
        except StopIteration:
            raise

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def timed_out(self) -> bool:
        if self._timeout is None or self._start_time is None:
            return False
        return time.monotonic() - self._start_time >= self._timeout

    @property
    def partial_output(self) -> str:
        return "".join(self._tokens)

    @property
    def token_count(self) -> int:
        return len(self._tokens)


class StreamingExecutor:
    """Wraps an executor to support token-level streaming."""

    def __init__(self, executor: Any) -> None:
        self._executor = executor
        self._callbacks: list[Callable[[str], None]] = []

    def add_callback(self, callback: Callable[[str], None]) -> None:
        self._callbacks.append(callback)

    def stream(
        self,
        task: dict[str, Any],
        timeout: float | None = None,
    ) -> TokenStream:
        if hasattr(self._executor, "stream_generate"):
            gen = self._executor.stream_generate(task)
        else:
            gen = self._default_generator(task)
        stream = TokenStream(gen, timeout=timeout)
        return stream

    def _default_generator(self, task: dict[str, Any]) -> Generator[str, None, None]:
        output = self._executor.execute(task)
        if isinstance(output, str):
            yield output
        else:
            yield str(output)

    def stream_with_callbacks(
        self,
        task: dict[str, Any],
        timeout: float | None = None,
    ) -> Generator[str, None, None]:
        stream = self.stream(task, timeout=timeout)
        for token in stream:
            for cb in self._callbacks:
                cb(token)
            yield token


class TokenCollector:
    """Collects streamed tokens into a complete response."""

    @staticmethod
    def collect(stream: TokenStream) -> str:
        for _ in stream:
            pass
        return stream.partial_output

    @staticmethod
    def collect_with_stats(stream: TokenStream) -> tuple[str, StreamStatistics]:
        start = time.perf_counter()
        for _ in stream:
            pass
        duration = (time.perf_counter() - start) * 1000.0
        stats = StreamStatistics(
            total_tokens=stream.token_count,
            total_duration_ms=duration,
            tokens_per_second=(stream.token_count / (duration / 1000.0)) if duration > 0 else 0.0,
            cancelled=stream.cancelled,
            timed_out=stream.timed_out,
            partial_output=stream.partial_output,
        )
        return stream.partial_output, stats


# Backward-compatible aliases
StreamAdapter = StreamingExecutor
TokenStreamCollector = TokenCollector
