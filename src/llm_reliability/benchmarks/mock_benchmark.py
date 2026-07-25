"""
Purpose
-------
Provide a deterministic, fully functional mock benchmark for testing the framework pipeline.

Responsibilities
----------------
- Implement the Benchmark interface
- Supply a small deterministic dataset without external dependencies
- Produce ExecutionRecords and EvaluationRecords consistently given the same seed
- Never use uncontrolled randomness

Usage example
-------------
>>> from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
>>> benchmark = MockBenchmark(seed=42)
>>> benchmark.load()
>>> tasks = benchmark.list_tasks()
>>> print(len(tasks))
10
"""

from __future__ import annotations

import hashlib
from typing import Any

from llm_reliability.configs.config import Configuration
from llm_reliability.interfaces.agent import Agent
from llm_reliability.interfaces.benchmark import Benchmark
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord

MOCK_TASKS = [
    {
        "task_id": f"mock-task-{i}",
        "prompt": f"Solve mock problem {i}.",
        "expected_answer": f"Answer {i}",
        "difficulty": "easy" if i < 5 else "hard",
        "category": "logic" if i % 2 == 0 else "math",
    }
    for i in range(10)
]


class MockBenchmark(Benchmark):
    """Deterministic mock benchmark for pipeline testing."""

    def __init__(
        self,
        config: Configuration | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize the MockBenchmark with a configuration or seed."""

        self._config = config
        self._seed = config.seed if config is not None else seed

        self._tasks: dict[str, Any] = {}
        self._logs: list[dict[str, Any]] = []
        self._loaded = False

    def load(self) -> None:
        """Load deterministic mock tasks."""
        self._tasks = {t["task_id"]: t for t in MOCK_TASKS}
        self._loaded = True
        self._logs.append({"event": "load", "status": "success", "task_count": len(self._tasks)})

    def list_tasks(self) -> list[str]:
        """Return all task identifiers in deterministic order."""
        if not self._loaded:
            raise RuntimeError("Benchmark not loaded.")
        return list(self._tasks.keys())

    def get_task(self, task_id: str) -> dict[str, Any]:
        """Return the task payload for the given identifier."""
        if not self._loaded:
            raise RuntimeError("Benchmark not loaded.")
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task_id: {task_id}")
        return self._tasks[task_id].copy()

    def run(self, agent: Agent, task: dict[str, Any]) -> ExecutionRecord:
        """Execute the agent on a task and return an ExecutionRecord."""
        # Deterministic timing
        h = hashlib.sha256(f"{self._seed}_{task['task_id']}".encode())
        deterministic_int = int(h.hexdigest()[:8], 16)
        runtime_seconds = 1.0 + (deterministic_int % 40) / 10.0
        timestamp = f"2026-01-01T00:{deterministic_int % 60:02d}:00+00:00"

        try:
            agent_output = agent.run(task)
            status = "success"
            error = None
        except Exception as e:
            agent_output = None
            status = "error"
            error = str(e)

        config_hash = hashlib.sha256(f"mock_config_{self._seed}".encode()).hexdigest()

        # Use the logical agent name from configuration (AgentSpec.name) so that
        # two agent specs backed by the same class (e.g. "mock" and "mock_agent")
        # are correctly distinguished in metric aggregation and ranking.
        agent_label = (
            self._config.agent
            if self._config is not None and self._config.agent
            else agent.__class__.__name__
        )

        record = ExecutionRecord(
            configuration_hash=config_hash,
            seed=self._seed,
            benchmark="MockBenchmark",
            agent=agent_label,
            task_id=task["task_id"],
            run_index=0,
            runtime_seconds=runtime_seconds,
            timestamp=timestamp,
            stdout="Mock output",
            stderr="",
            status=status,
            error=error,
            agent_output=agent_output,
            software_versions={"mock": "1.0"},
            environment_metadata={"mock_env": True},
        )

        self._logs.append(
            {
                "event": "run",
                "task_id": task["task_id"],
                "status": status,
                "runtime_seconds": runtime_seconds,
            }
        )

        return record

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        """Evaluate one execution deterministically and return an EvaluationRecord."""
        task = self.get_task(execution.task_id)

        h = hashlib.sha256(f"eval_{self._seed}_{execution.task_id}".encode())
        deterministic_int = int(h.hexdigest()[:8], 16)
        evaluated_at = f"2026-01-01T01:{deterministic_int % 60:02d}:00+00:00"

        agent_output = execution.agent_output
        expected = task.get("expected_answer")

        if execution.status == "error":
            success = False
            score = 0.0
        else:
            success = str(expected) == str(agent_output).strip()
            score = 1.0 if success else 0.0

        eval_record = EvaluationRecord.from_execution(
            execution=execution,
            success=success,
            score=score,
            metrics={"difficulty": task.get("difficulty")},
            evaluated_at=evaluated_at,
        )

        self._logs.append({"event": "evaluate", "task_id": execution.task_id, "success": success})

        return eval_record

    def collect_logs(self) -> dict[str, Any]:
        """Return benchmark-level logs accumulated during execution."""
        return {"logs": self._logs.copy()}

    def metadata(self) -> dict[str, Any]:
        """Return descriptive metadata about this benchmark adapter."""
        return {
            "name": "MockBenchmark",
            "version": "1.0",
            "deterministic": True,
            "task_count": len(self._tasks) if self._loaded else 0,
        }
