"""
Runtime — abstract interface for interchangeable inference backends.

``Runtime`` extends ``Agent`` with an explicit ``execute()`` entry-point
and optional capability methods (model loading/unloading, health checking,
token counting, latency/memory measurement, version detection).

Every method has a default implementation so existing agent subclasses
continue to work without modification.  Subclasses override the methods
they support; unsupported methods return safe defaults.
"""

from __future__ import annotations

import time
from abc import ABC
from typing import Any

from llm_reliability.interfaces.agent import Agent
from llm_reliability.runtime.metadata import RuntimeCapabilities, RuntimeMetadata


class Runtime(Agent, ABC):
    """Abstract interface for all inference runtimes.

    Core lifecycle
    --------------
    ``initialize()`` → ``execute()`` → ``shutdown()``

    Optional capabilities (default implementations do nothing / return None)
    ------------------------------------------------------------------------
    - ``load_model()`` / ``unload_model()``  – manage model lifecycle
    - ``health_check()``                     – verify runtime is responsive
    - ``count_tokens(text)``                 – tokenize without inference
    - ``measure_latency(task)``              – time a single inference
    - ``measure_memory()``                   – report current memory usage
    - ``runtime_metadata()``                 – return ``RuntimeMetadata``
    """

    def execute(self, task: dict[str, Any]) -> Any:
        """Execute a task and return the raw output.

        The default implementation delegates to ``run()`` for backward
        compatibility.
        """
        return self.run(task)

    # ------------------------------------------------------------------
    # Optional capability methods — default implementations are no-ops
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the model into memory (if runtime supports separate loading)."""

    def unload_model(self) -> None:
        """Unload the model from memory (if runtime supports separate unloading)."""

    def health_check(self) -> bool:
        """Return True if the runtime is reachable and responsive.

        Default returns True (assumes healthy) — override if the runtime
        provides an explicit health endpoint.
        """
        return True

    def count_tokens(self, text: str) -> int:
        """Return token count for *text* using the runtime's tokenizer.

        Default returns 0 — override when the runtime exposes token counting.
        """
        return 0

    def measure_latency(self, task: dict[str, Any]) -> tuple[Any, float]:
        """Execute *task* and return (output, latency_ms).

        Default calls ``execute()`` and returns (output, latency_ms).
        """
        start = time.perf_counter()
        output = self.execute(task)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return output, latency_ms

    def measure_memory(self) -> dict[str, float]:
        """Return current memory usage in MB.

        Default returns empty dict — override when the runtime exposes
        memory usage information.
        """
        return {}

    def runtime_metadata(self) -> RuntimeMetadata:
        """Return standardized metadata about this runtime instance.

        Default returns a minimal ``RuntimeMetadata`` with just the class
        name.  Override to populate version, backend, quantization, etc.
        """
        return RuntimeMetadata(
            runtime_name=self.__class__.__name__,
        )

    def _detect_capabilities(self) -> RuntimeCapabilities:
        """Check which optional methods have been overridden in a subclass.

        Uses function ``__code__`` identity comparison against the default
        implementation on ``Runtime``.
        """
        is_overridden = {}
        checks = [
            ("load_model", "model_loading"),
            ("unload_model", "model_unloading"),
            ("health_check", "health_check"),
            ("count_tokens", "token_counting"),
            ("measure_memory", "memory_measurement"),
        ]
        for method_name, cap_name in checks:
            default = Runtime.__dict__.get(method_name)
            current = self.__class__.__dict__.get(method_name)
            if default is None or current is None:
                is_overridden[cap_name] = current is not None
            else:
                is_overridden[cap_name] = getattr(current, "__code__", None) != getattr(
                    default, "__code__", None
                )
        return RuntimeCapabilities(**is_overridden)
