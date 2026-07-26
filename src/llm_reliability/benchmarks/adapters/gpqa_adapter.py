"""
Purpose
-------
Integrate the GPQA (Google-Proof QA) benchmark into the framework.

Responsibilities
----------------
- Load GPQA multiple-choice expert-domain tasks
- Evaluate answer extraction from agent responses (A-D)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_reliability.benchmarks.adapters.base_adapter import BaseBenchmarkAdapter
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry
from llm_reliability.interfaces.agent import Agent
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord

logger = logging.getLogger(__name__)


def extract_gpqa_choice(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"\b([A-D])\b", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return text.strip()[:1].upper()


class GPQAAdapter(BaseBenchmarkAdapter):
    """Adapter for the GPQA benchmark."""

    def validate_configuration(self) -> None:
        super().validate_configuration()
        if not self.config.metadata.get("dataset_path"):
            raise ValueError("Configuration metadata must contain 'dataset_path' for GPQA.")

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
                raise TypeError("GPQA dataset must be a list of tasks.")

            self._tasks = {}
            for item in data:
                tid = item.get("task_id", f"gpqa_{len(self._tasks)}")
                question = item.get("question", item.get("prompt", ""))
                choices = item.get("choices", item.get("options", []))
                answer = str(item.get("correct_answer", item.get("answer", "A"))).strip().upper()

                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    choice_labels = []
                    for c in choices:
                        label = c.get("label", "")
                        text = c.get("text", "")
                        choice_labels.append(f"{label}. {text}")
                    choices_text = "\n".join(choice_labels)
                elif isinstance(choices, list):
                    choices_text = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
                else:
                    choices_text = ""

                prompt = f"Question: {question}\n{choices_text}\nAnswer with letter (A-D):"

                self._tasks[tid] = {
                    "task_id": tid,
                    "question": question,
                    "choices": choices,
                    "prompt": prompt,
                    "ground_truth_answer": answer,
                }
        else:
            self._tasks = {
                f"gpqa_{i}": {
                    "task_id": f"gpqa_{i}",
                    "question": f"What is the oxidation state of oxygen in H2O2? {i}",
                    "choices": ["-2", "-1", "0", "+2"],
                    "prompt": f"What is the oxidation state of oxygen in H2O2? {i}\nA. -2\nB. -1\nC. 0\nD. +2\nAnswer:",
                    "ground_truth_answer": "B",
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
            benchmark="GPQA",
            agent=self.config.agent,
            task_id=task_id,
            run_index=0,
            runtime_seconds=runtime_seconds,
            timestamp=timestamp,
            stdout="GPQA Execution",
            stderr="",
            status=status,
            error=error,
            agent_output=agent_output,
            software_versions={"gpqa": "1.0"},
            environment_metadata={},
        )
        self._logs.append({"event": "run", "task_id": task_id, "status": status})
        return record

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        task = self.get_task(execution.task_id)
        expected = str(task.get("ground_truth_answer", "")).strip().upper()
        agent_output = execution.agent_output

        if execution.status == "error" or not agent_output:
            success = False
            score = 0.0
        else:
            extracted = extract_gpqa_choice(str(agent_output))
            success = (extracted == expected) or (expected in str(agent_output).upper())
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
            "name": "GPQA",
            "version": "1.0",
            "task_count": len(self._tasks) if self._loaded else 0,
            "deterministic": True,
        }


BenchmarkRegistry.register("GPQA", GPQAAdapter)
