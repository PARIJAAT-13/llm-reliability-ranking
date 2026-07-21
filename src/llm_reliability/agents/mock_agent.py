from __future__ import annotations

from typing import Any

from llm_reliability.configs.config import Configuration
from llm_reliability.interfaces.agent import Agent


class MockAgent(Agent):
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