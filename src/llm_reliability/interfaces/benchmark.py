"""
Purpose
-------
Define the abstract benchmark contract that all benchmark adapters must
implement without modifying framework code.

Responsibilities
----------------
- Load benchmark resources and enumerate tasks
- Orchestrate agent execution on individual tasks
- Own evaluation logic and produce EvaluationRecords
- Expose benchmark metadata and execution logs

Usage example
-------------
>>> from llm_reliability.interfaces import Benchmark
>>> class MyBenchmark(Benchmark):
...     def load(self) -> None: ...
...     # implement remaining abstract methods
>>> isinstance(MyBenchmark(), Benchmark)
True

Design notes
------------
Benchmarks sit between Configuration and Agent in the dependency graph.
The ``run`` method receives an Agent instance and produces an
ExecutionRecord; ``evaluate`` converts that record into an EvaluationRecord.
Benchmarks never depend on metrics, rankings, or statistics modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_reliability.interfaces.agent import Agent
    from llm_reliability.records.evaluation import EvaluationRecord
    from llm_reliability.records.execution import ExecutionRecord


class Benchmark(ABC):
    """Abstract interface for all benchmark adapters."""

    @abstractmethod
    def load(self) -> None:
        """Load benchmark resources such as datasets and evaluators."""

    @abstractmethod
    def list_tasks(self) -> list[str]:
        """Return all task identifiers in deterministic order."""

    @abstractmethod
    def get_task(self, task_id: str) -> dict[str, Any]:
        """Return the task payload for the given identifier."""

    @abstractmethod
    def run(self, agent: Agent, task: dict[str, Any]) -> ExecutionRecord:
        """Execute the agent on a task and return an ExecutionRecord."""

    @abstractmethod
    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        """Evaluate one execution and return an EvaluationRecord."""

    @abstractmethod
    def collect_logs(self) -> dict[str, Any]:
        """Return benchmark-level logs accumulated during execution."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return descriptive metadata about this benchmark adapter."""
