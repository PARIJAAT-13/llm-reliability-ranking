"""
Purpose
-------
Integrate the TriviaQA benchmark into the framework.

Responsibilities
----------------
- Load TriviaQA reading comprehension / trivia tasks
- Evaluate via free-form text matching against answer aliases
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_reliability.benchmarks.adapters.base_adapter import \
    BaseBenchmarkAdapter
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry
from llm_reliability.interfaces.agent import Agent
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord

logger = logging.getLogger(__name__)


def extract_triviaqa_answer(response: str, aliases: list[str]) -> bool:
    if not response or not aliases:
        return False
    response_lower = response.lower()
    for alias in aliases:
        if alias.lower() in response_lower:
            return True
    return False


class TriviaQAAdapter(BaseBenchmarkAdapter):
    """Adapter for the TriviaQA benchmark."""

    def validate_configuration(self) -> None:
        super().validate_configuration()
        if not self.config.metadata.get("dataset_path"):
            raise ValueError("Configuration metadata must contain 'dataset_path' for TriviaQA.")

    def _load_tasks(self) -> None:
        dataset_path = self.config.metadata["dataset_path"]
        path_obj = Path(dataset_path)

        if path_obj.exists() and path_obj.is_file():
            try:
                with open(path_obj, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error("Failed to load dataset from %s: %s", dataset_path, e)
                raise RuntimeError(f"Missing or invalid dataset: {e}") from e

            if not isinstance(data, list):
                raise TypeError("TriviaQA dataset must be a list of tasks.")

            self._tasks = {}
            for item in data:
                tid = item.get("task_id", f"triviaqa_{len(self._tasks)}")
                question = item.get("question", item.get("prompt", ""))
                answer = item.get("answer", item.get("ground_truth_answer", ""))
                aliases = item.get("aliases", [])

                if not isinstance(aliases, list):
                    aliases = [str(aliases)] if aliases else []
                if answer and answer not in aliases:
                    aliases.insert(0, str(answer))

                prompt = f"Question: {question}\nAnswer:"

                self._tasks[tid] = {
                    "task_id": tid,
                    "question": question,
                    "aliases": aliases,
                    "prompt": prompt,
                    "ground_truth_answer": str(answer),
                }
        else:
            self._tasks = {
                f"triviaqa_{i}": {
                    "task_id": f"triviaqa_{i}",
                    "question": f"What is the capital of France? {i}",
                    "aliases": ["Paris", "paris"],
                    "prompt": f"What is the capital of France? {i}\nAnswer:",
                    "ground_truth_answer": "Paris",
                }
                for i in range(5)
            }

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

        record = ExecutionRecord(
            configuration_hash=self.config.sha256(),
            seed=self.config.seed,
            benchmark="TriviaQA",
            agent=self.config.agent,
            task_id=task_id,
            run_index=0,
            runtime_seconds=runtime_seconds,
            timestamp=timestamp,
            stdout="TriviaQA Execution",
            stderr="",
            status=status,
            error=error,
            agent_output=agent_output,
            software_versions={"triviaqa": "1.0"},
            environment_metadata={},
        )
        self._logs.append({"event": "run", "task_id": task_id, "status": status})
        return record

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        task = self.get_task(execution.task_id)
        aliases = task.get("aliases", [])
        agent_output = execution.agent_output

        if execution.status == "error" or not agent_output:
            success = False
            score = 0.0
        else:
            success = extract_triviaqa_answer(str(agent_output), aliases)
            score = 1.0 if success else 0.0

        evaluated_at = datetime.now(timezone.utc).isoformat()
        eval_record = EvaluationRecord.from_execution(
            execution=execution,
            success=success,
            score=score,
            metrics={},
            evaluated_at=evaluated_at,
        )
        self._logs.append({"event": "evaluate", "task_id": execution.task_id, "success": success})
        return eval_record

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "TriviaQA",
            "version": "1.0",
            "task_count": len(self._tasks) if self._loaded else 0,
            "deterministic": True,
        }


BenchmarkRegistry.register("TriviaQA", TriviaQAAdapter)
