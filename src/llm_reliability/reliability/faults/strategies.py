"""
Purpose
-------
Provide concrete fault injection strategies for LLM agent fault tolerance evaluation.

Responsibilities
----------------
- Implement ArtificialTimeoutFaultStrategy
- Implement TemporaryApiFailureFaultStrategy
- Implement InvalidModelResponseFaultStrategy
- Implement ToolFailureFaultStrategy
- Implement ContextTruncationFaultStrategy
- Implement NetworkInterruptionFaultStrategy

Design notes
------------
All strategies inherit from FaultInjectionStrategy.
They simulate controlled failure modes at specific injection points during execution,
allowing the FaultManager to record recovery behaviors and retry telemetry.
"""

from __future__ import annotations

import random
import time
from typing import Any

from llm_reliability.reliability.faults.base import FaultInjectionStrategy


class ArtificialTimeoutFaultStrategy(FaultInjectionStrategy):
    """Delay execution to simulate artificial timeouts."""

    def __init__(self, delay_seconds: float = 0.5, raise_timeout: bool = False) -> None:
        self.delay_seconds = delay_seconds
        self.raise_timeout = raise_timeout

    @property
    def fault_name(self) -> str:
        return "artificial_timeout"

    @property
    def injection_point(self) -> str:
        return "agent_run"

    @property
    def description(self) -> str:
        return "Inject artificial execution delay or trigger timeout exception."

    def inject(self, target: Any, seed: int | None = None, **kwargs: Any) -> Any:
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

        if self.raise_timeout:
            raise TimeoutError("Artificial execution timeout exceeded.")

        return target

    def cleanup(self) -> None:
        pass


class TemporaryApiFailureFaultStrategy(FaultInjectionStrategy):
    """Simulate transient service failures that succeed upon retry."""

    def __init__(self, max_failures: int = 1) -> None:
        self.max_failures = max_failures
        self.current_failures = 0

    @property
    def fault_name(self) -> str:
        return "temporary_api_failure"

    @property
    def injection_point(self) -> str:
        return "api_call"

    @property
    def description(self) -> str:
        return "Simulate transient service failure (503 Service Unavailable)."

    def inject(self, target: Any, seed: int | None = None, **kwargs: Any) -> Any:
        if self.current_failures < self.max_failures:
            self.current_failures += 1
            raise RuntimeError("503 Service Unavailable: Transient API failure.")
        return target

    def cleanup(self) -> None:
        self.current_failures = 0


class InvalidModelResponseFaultStrategy(FaultInjectionStrategy):
    """Inject empty, malformed JSON, or unexpected data type responses."""

    def __init__(self, mode: str = "empty") -> None:
        self.mode = mode

    @property
    def fault_name(self) -> str:
        return "invalid_model_response"

    @property
    def injection_point(self) -> str:
        return "agent_run"

    @property
    def description(self) -> str:
        return "Inject empty output, malformed JSON, or unexpected data type."

    def inject(self, target: Any, seed: int | None = None, **kwargs: Any) -> Any:
        rng = random.Random(seed if seed is not None else 42)
        effective_mode = self.mode
        if effective_mode == "random":
            effective_mode = rng.choice(["empty", "malformed_json", "unexpected_type"])

        if effective_mode == "empty":
            return ""
        elif effective_mode == "malformed_json":
            return '{"status": "error", "data": '
        elif effective_mode == "unexpected_type":
            return 12345
        return ""

    def cleanup(self) -> None:
        pass


class ToolFailureFaultStrategy(FaultInjectionStrategy):
    """Simulate unavailable tool or tool execution exception."""

    def __init__(self, tool_name: str = "web_search") -> None:
        self.tool_name = tool_name

    @property
    def fault_name(self) -> str:
        return "tool_failure"

    @property
    def injection_point(self) -> str:
        return "tool_call"

    @property
    def description(self) -> str:
        return "Simulate unavailable tool or tool execution exception."

    def inject(self, target: Any, seed: int | None = None, **kwargs: Any) -> Any:
        raise RuntimeError(f"Tool execution error: Tool '{self.tool_name}' is unavailable.")

    def cleanup(self) -> None:
        pass


class ContextTruncationFaultStrategy(FaultInjectionStrategy):
    """Remove part of prompt text deterministically."""

    def __init__(self, truncation_ratio: float = 0.3) -> None:
        self.truncation_ratio = min(max(truncation_ratio, 0.0), 0.9)

    @property
    def fault_name(self) -> str:
        return "context_truncation"

    @property
    def injection_point(self) -> str:
        return "prompt"

    @property
    def description(self) -> str:
        return "Truncate a portion of the task prompt while preserving task identity."

    def inject(self, target: Any, seed: int | None = None, **kwargs: Any) -> Any:
        if isinstance(target, dict):
            task_copy = target.copy()
            for key in ("prompt", "instruction", "query", "task"):
                val = task_copy.get(key)
                if isinstance(val, str) and val.strip():
                    keep_len = max(1, int(len(val) * (1.0 - self.truncation_ratio)))
                    task_copy[key] = val[:keep_len]
                    break
            return task_copy
        elif isinstance(target, str):
            keep_len = max(1, int(len(target) * (1.0 - self.truncation_ratio)))
            return target[:keep_len]
        return target

    def cleanup(self) -> None:
        pass


class NetworkInterruptionFaultStrategy(FaultInjectionStrategy):
    """Simulate recoverable network connection error."""

    @property
    def fault_name(self) -> str:
        return "network_interruption"

    @property
    def injection_point(self) -> str:
        return "api_call"

    @property
    def description(self) -> str:
        return "Simulate recoverable network connection reset error."

    def inject(self, target: Any, seed: int | None = None, **kwargs: Any) -> Any:
        raise ConnectionResetError("Connection reset by peer during API request.")

    def cleanup(self) -> None:
        pass
