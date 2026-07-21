"""
Purpose
-------
Define the abstract agent contract that all LLM agent adapters must implement.

Responsibilities
----------------
- Initialize and tear down agent runtime resources
- Reset state between task executions
- Execute tasks and return raw agent outputs
- Expose agent metadata for logging and reproducibility

Usage example
-------------
>>> from llm_reliability.interfaces import Agent
>>> class MyAgent(Agent):
...     def initialize(self) -> None: ...
...     # implement remaining abstract methods
>>> isinstance(MyAgent(), Agent)
True

Design notes
------------
Agents never evaluate themselves. Evaluation is exclusively owned by the
Benchmark. The Agent ``run`` method returns the raw output payload; the
Benchmark wraps it into an ExecutionRecord with full logging context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """Abstract interface for all LLM agent adapters."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize agent runtime resources."""

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state between task executions."""

    @abstractmethod
    def run(self, task: dict[str, Any]) -> Any:
        """Execute the agent on a task and return the raw output."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release agent runtime resources."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return descriptive metadata about this agent adapter."""
