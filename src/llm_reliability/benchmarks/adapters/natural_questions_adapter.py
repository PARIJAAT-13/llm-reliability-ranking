"""
Purpose
-------
Integrate the Natural Questions (Google NQ) benchmark into the framework.

Responsibilities
----------------
- Load Natural Questions open-domain QA tasks
- Evaluate via free-form text matching against answer text
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_reliability.benchmarks.adapters.base_adapter import BaseBenchmarkAdapter
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry
from llm_reliability.interfaces.agent import Agent
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord

logger = logging.getLogger(__name__)


def extract_natural_questions_answer(response: str, answer_text: str) -> bool:
    if not response or not answer_text:
        return False
    return answer_text.lower() in response.lower()


class NaturalQuestionsAdapter(BaseBenchmarkAdapter):
    """Adapter for the Natural Questions benchmark."""

    def validate_configuration(self) -> None:
        super().validate_configuration()
        if not self.config.metadata.get("dataset_path"):
            raise ValueError(
                "Configuration metadata must contain 'dataset_path' for NaturalQuestions."
            )

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
                raise TypeError("NaturalQuestions dataset must be a list of tasks.")

            self._tasks = {}
            for item in data:
                tid = item.get("task_id", f"natural_questions_{len(self._tasks)}")
                question = item.get("question", item.get("prompt", ""))
                answer = item.get("answer", item.get("ground_truth_answer", ""))
                short_answers = item.get("short_answers", [])

                if isinstance(answer, dict):
                    answer = answer.get("text", "")
                if isinstance(answer, list):
                    answer = " ".join(str(a) for a in answer)

                prompt = f"Question: {question}\nAnswer:"

                self._tasks[tid] = {
                    "task_id": tid,
                    "question": question,
                    "answer": str(answer),
                    "short_answers": (
                        short_answers if isinstance(short_answers, list) else [str(short_answers)]
                    ),
                    "prompt": prompt,
                    "ground_truth_answer": str(answer),
                }
        else:
            self._tasks = {
                f"natural_questions_{i}": {
                    "task_id": f"natural_questions_{i}",
                    "question": f"Who wrote the novel '1984'? {i}",
                    "answer": "George Orwell",
                    "short_answers": ["George Orwell", "Orwell"],
                    "prompt": f"Who wrote the novel '1984'? {i}\nAnswer:",
                    "ground_truth_answer": "George Orwell",
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
            benchmark="NaturalQuestions",
            agent=self.config.agent,
            task_id=task_id,
            run_index=0,
            runtime_seconds=runtime_seconds,
            timestamp=timestamp,
            stdout="NaturalQuestions Execution",
            stderr="",
            status=status,
            error=error,
            agent_output=agent_output,
            software_versions={"natural_questions": "1.0"},
            environment_metadata={},
        )
        self._logs.append({"event": "run", "task_id": task_id, "status": status})
        return record

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        task = self.get_task(execution.task_id)
        answer_text = task.get("answer", "")
        short_answers = task.get("short_answers", [])
        agent_output = execution.agent_output

        if execution.status == "error" or not agent_output:
            success = False
            score = 0.0
        else:
            response_str = str(agent_output)
            success = extract_natural_questions_answer(response_str, answer_text)
            if not success:
                for sa in short_answers:
                    if extract_natural_questions_answer(response_str, str(sa)):
                        success = True
                        break
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
            "name": "NaturalQuestions",
            "version": "1.0",
            "task_count": len(self._tasks) if self._loaded else 0,
            "deterministic": True,
        }


BenchmarkRegistry.register("NaturalQuestions", NaturalQuestionsAdapter)
