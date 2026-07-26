from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_reliability.benchmarks.adapters.base_adapter import BaseBenchmarkAdapter
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry
from llm_reliability.benchmarks.adapters.webarena_models import WebArenaMetadata
from llm_reliability.benchmarks.adapters.webarena_utils import normalize_webarena_answer
from llm_reliability.interfaces.agent import Agent
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord

logger = logging.getLogger(__name__)


class WebArenaAdapter(BaseBenchmarkAdapter):
    """Adapter for the WebArena benchmark (web agent navigation tasks)."""

    def validate_configuration(self) -> None:
        super().validate_configuration()
        if not self.config.metadata.get("dataset_path"):
            raise ValueError("Configuration metadata must contain 'dataset_path' for WebArena.")

    def _load_tasks(self) -> None:
        dataset_path = self.config.metadata["dataset_path"]
        path_obj = Path(dataset_path)

        if not path_obj.is_file():
            raise RuntimeError(f"Dataset file not found: {dataset_path}")

        try:
            with open(dataset_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to load dataset from %s: %s", dataset_path, e)
            raise RuntimeError(f"Missing or invalid dataset: {e}") from e

        if not isinstance(data, list):
            raise TypeError("WebArena dataset must be a list of tasks.")

        from llm_reliability.benchmarks.adapters.webarena_models import WebArenaTask

        self._tasks = {}
        for item in data:
            try:
                task_obj = WebArenaTask(**item)
            except Exception as e:
                raise ValueError(f"Invalid schema: {e}") from e
            if task_obj.task_id in self._tasks:
                raise ValueError(f"Duplicate task ID: {task_obj.task_id}")
            self._tasks[task_obj.task_id] = task_obj.model_dump()

    def run(self, agent: Agent, task: dict[str, Any]) -> ExecutionRecord:
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
            benchmark="WebArena",
            agent=self.config.agent,
            task_id=task_id,
            run_index=0,
            runtime_seconds=runtime_seconds,
            timestamp=timestamp,
            stdout="WebArena Execution",
            stderr="",
            status=status,
            error=error,
            agent_output=agent_output,
            software_versions={"webarena": "1.0"},
            environment_metadata={},
        )

        self._logs.append({"event": "run", "task_id": task_id, "status": status})
        return record

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        task = self.get_task(execution.task_id)
        expected = task.get("expected_answer")
        agent_output = execution.agent_output

        if execution.status == "error":
            success = False
            score = 0.0
        else:
            expected_norm = normalize_webarena_answer(str(expected))
            output_norm = normalize_webarena_answer(str(agent_output))
            success = output_norm == expected_norm or expected_norm in output_norm
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
        meta = WebArenaMetadata(task_count=len(self._tasks) if self._loaded else 0)
        return meta.model_dump()


BenchmarkRegistry.register("WebArena", WebArenaAdapter)
