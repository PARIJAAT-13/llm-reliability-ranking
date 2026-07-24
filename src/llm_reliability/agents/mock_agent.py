from __future__ import annotations

from typing import Any

from llm_reliability.configs.config import Configuration
from llm_reliability.runtime import Runtime
from llm_reliability.runtime.registry import RuntimeRegistry


class MockAgent(Runtime):
    """Simple deterministic mock agent."""

    def __init__(self, config: Configuration | None = None):
        self._config = config

    def initialize(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def run(self, task: dict[str, Any]) -> Any:
        return task.get("expected_answer", "")

    def shutdown(self) -> None:
        pass

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "MockAgent",
            "version": "1.0",
            "deterministic": True,
        }


if not RuntimeRegistry.exists("mock"):
    RuntimeRegistry.register("mock", MockAgent)
