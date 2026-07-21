"""
Purpose
-------
Integrate the AgentBoard benchmark into the LLM Reliability Ranking framework.

Responsibilities
----------------
- Load and parse JSON dataset files for AgentBoard
- Convert AgentBoard scenarios into framework ExecutionRecords
- Evaluate the execution success deterministically
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from llm_reliability.benchmarks.adapters.agentboard_models import (
    AgentBoardMetadata,
    AgentBoardTask,
)
from llm_reliability.benchmarks.adapters.base_adapter import BaseBenchmarkAdapter
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry
from llm_reliability.interfaces.agent import Agent
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord

logger = logging.getLogger(__name__)


class AgentBoardAdapter(BaseBenchmarkAdapter):
    """Adapter for the AgentBoard benchmark."""

    def validate_configuration(self) -> None:
        """Validate AgentBoard specific configuration."""
        super().validate_configuration()
        if not self.config.metadata.get("dataset_path"):
            raise ValueError(
                "Configuration metadata must contain 'dataset_path' for AgentBoard."
            )

    def _load_tasks(self) -> None:
        """Load and validate the AgentBoard dataset."""
        dataset_path = self.config.metadata["dataset_path"]
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to load dataset from %s: %s", dataset_path, e)
            raise RuntimeError(f"Missing or invalid dataset: {e}") from e

        if not isinstance(data, list):
            raise TypeError("AgentBoard dataset must be a list of tasks.")

        self._tasks = {}
        for item in data:
            try:
                task_obj = AgentBoardTask(**item)
                if task_obj.task_id in self._tasks:
                    raise ValueError(f"Duplicate task ID found: {task_obj.task_id}")
                self._tasks[task_obj.task_id] = task_obj.model_dump()
            except Exception as e:
                logger.error("Malformed task in dataset: %s", e)
                raise ValueError(f"Invalid schema: {e}") from e

    def run(self, agent: Agent, task: dict[str, Any]) -> ExecutionRecord:
        """Execute a single AgentBoard task using the provided agent."""
        task_id = task["task_id"]

        start_time = datetime.now(timezone.utc)
        try:
            agent_output = agent.run(task)
            status = "success"
            error = None
        except Exception as e:
            agent_output = None
            status = "error"
            error = str(e)

        # Deterministic timing for tests
        if self.config.seed is not None:
            h = hashlib.sha256(f"{self.config.seed}_{task_id}".encode())
            deterministic_int = int(h.hexdigest()[:8], 16)
            runtime_seconds = 1.0 + (deterministic_int % 40) / 10.0
            timestamp = f"2026-01-01T00:{deterministic_int % 60:02d}:00+00:00"
        else:
            runtime_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
            timestamp = start_time.isoformat()

        config_hash = self.config.sha256()

        record = ExecutionRecord(
            configuration_hash=config_hash,
            seed=self.config.seed,
            benchmark="AgentBoard",
            # Use the logical agent name from config (AgentSpec.name) not the class name.
            agent=self.config.agent,
            task_id=task_id,
            run_index=0,
            runtime_seconds=runtime_seconds,
            timestamp=timestamp,
            stdout="AgentBoard Execution",
            stderr="",
            status=status,
            error=error,
            agent_output=agent_output,
            software_versions={"agentboard": "1.0"},
            environment_metadata={},
        )
        
        self._logs.append({"event": "run", "task_id": task_id, "status": status})
        return record

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        """Evaluate the agent output against AgentBoard expected output."""
        task = self.get_task(execution.task_id)
        expected = task.get("expected_output")
        agent_output = execution.agent_output

        if execution.status == "error":
            success = False
            score = 0.0
        else:
            success = str(expected).strip() == str(agent_output).strip()
            score = 1.0 if success else 0.0

        if self.config.seed is not None:
            h = hashlib.sha256(f"eval_{self.config.seed}_{execution.task_id}".encode())
            deterministic_int = int(h.hexdigest()[:8], 16)
            evaluated_at = f"2026-01-01T01:{deterministic_int % 60:02d}:00+00:00"
        else:
            evaluated_at = datetime.now(timezone.utc).isoformat()

        eval_record = EvaluationRecord.from_execution(
            execution=execution,
            success=success,
            score=score,
            metrics={"difficulty": task.get("difficulty")},
            evaluated_at=evaluated_at,
        )
        
        self._logs.append({"event": "evaluate", "task_id": execution.task_id, "success": success})
        return eval_record

    def metadata(self) -> dict[str, Any]:
        """Return AgentBoard metadata."""
        meta = AgentBoardMetadata(task_count=len(self._tasks) if self._loaded else 0)
        return meta.model_dump()


BenchmarkRegistry.register("AgentBoard", AgentBoardAdapter)
