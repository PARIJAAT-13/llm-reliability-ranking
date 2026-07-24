"""
Purpose
-------
Provide an abstract base class that simplifies benchmark integration.

Responsibilities
----------------
- Inherit from the Benchmark interface
- Provide reusable implementations for logging and metadata
- Enforce standard dataset and configuration validation hooks
- Leave actual execution and evaluation to subclasses
"""

from abc import abstractmethod
from typing import Any

from llm_reliability.benchmarks import BenchmarkPlugin
from llm_reliability.configs.config import Configuration
from llm_reliability.interfaces.agent import Agent
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord


class BaseBenchmarkAdapter(BenchmarkPlugin):
    """Abstract base adapter for real-world benchmark integrations."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the adapter with a framework configuration."""
        self.config = config
        self._logs: list[dict[str, Any]] = []
        self._tasks: dict[str, Any] = {}
        self._loaded = False
        self.validate_configuration()

    def validate_configuration(self) -> None:
        """
        Hook for subclasses to validate configuration parameters.
        Base implementation ensures a configuration exists.
        """
        if not self.config:
            raise ValueError("Configuration must be provided.")

    def validate_dataset(self) -> None:
        """
        Hook for subclasses to validate loaded dataset integrity.
        Base implementation ensures tasks dictionary is not empty.
        """
        if not self._tasks:
            raise ValueError("Dataset is empty. No tasks were loaded.")

    def load(self) -> None:
        """
        Load benchmark resources. Subclasses should override `_load_tasks`.
        This wrapper guarantees validation is applied.
        """
        self._load_tasks()
        self.validate_dataset()
        self._loaded = True
        self._logs.append({"event": "load", "task_count": len(self._tasks)})

    @abstractmethod
    def _load_tasks(self) -> None:
        """Subclass implementation to load tasks into self._tasks."""
        pass

    def list_tasks(self) -> list[str]:
        """Return all task identifiers in deterministic order."""
        if not self._loaded:
            raise RuntimeError("Benchmark not loaded.")
        return sorted(list(self._tasks.keys()))

    def get_task(self, task_id: str) -> dict[str, Any]:
        """Return the task payload for the given identifier."""
        if not self._loaded:
            raise RuntimeError("Benchmark not loaded.")
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task_id: {task_id}")
        return self._tasks[task_id].copy()

    @abstractmethod
    def run(self, agent: Agent, task: dict[str, Any]) -> ExecutionRecord:
        """Execute the agent on a task and return an ExecutionRecord."""
        pass

    @abstractmethod
    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        """Evaluate one execution and return an EvaluationRecord."""
        pass

    def collect_logs(self) -> dict[str, Any]:
        """Return benchmark-level logs accumulated during execution."""
        return {"logs": self._logs.copy()}

    def metadata(self) -> dict[str, Any]:
        """Return descriptive metadata about this benchmark adapter."""
        return {
            "name": self.__class__.__name__,
            "deterministic": True,
            "task_count": len(self._tasks) if self._loaded else 0,
        }
