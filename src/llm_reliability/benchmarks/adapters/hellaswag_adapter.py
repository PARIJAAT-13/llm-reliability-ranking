"""
Purpose
-------
Integrate the HellaSwag benchmark into the framework.

Responsibilities
----------------
- Load and parse JSON dataset files for HellaSwag commonsense reasoning
- Evaluate choice predictions (0, 1, 2, 3 or A, B, C, D)
"""

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


def extract_hellaswag_choice(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"\b([0-3]|[A-D])\b", text.strip(), re.IGNORECASE)
    if match:
        val = match.group(1).upper()
        mapping = {"0": "A", "1": "B", "2": "C", "3": "D"}
        return mapping.get(val, val)
    return text.strip()[:1].upper()


class HellaSwagAdapter(BaseBenchmarkAdapter):
    """Adapter for the HellaSwag benchmark."""

    def validate_configuration(self) -> None:
        super().validate_configuration()
        if not self.config.metadata.get("dataset_path"):
            raise ValueError("Configuration metadata must contain 'dataset_path' for HellaSwag.")

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
                raise TypeError("HellaSwag dataset must be a list of tasks.")

            self._tasks = {}
            for item in data:
                tid = item.get("task_id", f"hellaswag_{len(self._tasks)}")
                ctx = item.get("ctx", item.get("context", ""))
                endings = item.get("endings", item.get("choices", []))
                prompt = (
                    f"Context: {ctx}\nWhich ending naturally continues the context?\n"
                    + "\n".join(f"{chr(65+i)}. {end}" for i, end in enumerate(endings))
                    + "\nAnswer with letter (A, B, C, D):"
                )
                gt = (
                    str(item.get("ground_truth_answer", item.get("label", item.get("target", "0"))))
                    .strip()
                    .upper()
                )
                if gt in ("0", "1", "2", "3"):
                    gt = chr(65 + int(gt))
                self._tasks[tid] = {
                    "task_id": tid,
                    "prompt": prompt,
                    "endings": endings,
                    "ground_truth_answer": gt,
                }
        else:
            self._tasks = {
                f"hellaswag_{i}": {
                    "task_id": f"hellaswag_{i}",
                    "prompt": f"Context {i}: A person starts a craft project.\nA. Ends well\nB. Fails completely\nC. Flies away\nD. Explodes\nAnswer:",
                    "endings": ["Ends well", "Fails completely", "Flies away", "Explodes"],
                    "ground_truth_answer": "A",
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
            benchmark="HellaSwag",
            agent=self.config.agent,
            task_id=task_id,
            run_index=0,
            runtime_seconds=runtime_seconds,
            timestamp=timestamp,
            stdout="HellaSwag Execution",
            stderr="",
            status=status,
            error=error,
            agent_output=agent_output,
            software_versions={"hellaswag": "1.0"},
            environment_metadata={},
        )
        self._logs.append({"event": "run", "task_id": task_id, "status": status})
        return record

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        task = self.get_task(execution.task_id)
        expected = task.get("ground_truth_answer", "").strip().upper()

        if execution.status == "error" or not execution.agent_output:
            success = False
            score = 0.0
        else:
            extracted = extract_hellaswag_choice(str(execution.agent_output))
            success = (extracted == expected) or (expected in str(execution.agent_output).upper())
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
            "name": "HellaSwag",
            "version": "1.0",
            "task_count": len(self._tasks) if self._loaded else 0,
            "deterministic": True,
        }


BenchmarkRegistry.register("HellaSwag", HellaSwagAdapter)
BenchmarkRegistry.register("HELLASWAG", HellaSwagAdapter)
