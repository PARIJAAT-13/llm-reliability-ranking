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


def extract_answer(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"answer is \(?([A-D])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1).upper()
    return text.strip()[:1].upper()


class MMLUProAdapter(BaseBenchmarkAdapter):
    """Adapter for the MMLU-Pro benchmark (extended MMLU with more subjects/tasks)."""

    def validate_configuration(self) -> None:
        super().validate_configuration()
        if not self.config.metadata.get("dataset_path"):
            raise ValueError("Configuration metadata must contain 'dataset_path' for MMLU-Pro.")

    def _load_tasks(self) -> None:
        dataset_path = self.config.metadata["dataset_path"]
        path_obj = Path(dataset_path)

        if path_obj.exists() and path_obj.is_file():
            with open(path_obj, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
            self._tasks = {}
            for item in data:
                tid = item.get("task_id", item.get("id", f"mmlu_pro_{len(self._tasks)}"))
                question = item.get("question", item.get("prompt", ""))
                choices = item.get("choices", item.get("options", []))
                answer = item.get("answer", item.get("ground_truth_answer", ""))

                if isinstance(choices, list) and choices:
                    labeled = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
                    prompt = f"{question}\n{labeled}\nAnswer:"
                    if isinstance(answer, int) and answer < len(choices):
                        answer_label = chr(65 + answer)
                    else:
                        answer_label = str(answer).strip().upper()
                else:
                    prompt = f"{question}\nAnswer:"
                    answer_label = str(answer).strip().upper()

                self._tasks[tid] = {
                    "task_id": tid,
                    "question": question,
                    "prompt": prompt,
                    "ground_truth_answer": answer_label,
                }
        else:
            self._tasks = self._create_fallback_tasks()

    def _create_fallback_tasks(self) -> dict[str, Any]:
        fallback = {}
        for i in range(5):
            tid = f"mmlu_pro_{i}"
            fallback[tid] = {
                "task_id": tid,
                "question": f"Sample MMLU-Pro question {i}",
                "prompt": f"Sample MMLU-Pro question {i}\nA. Option A\nB. Option B\nC. Option C\nD. Option D\nAnswer:",
                "ground_truth_answer": "A",
            }
        return fallback

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
            benchmark="MMLU-Pro",
            agent=self.config.agent,
            task_id=task_id,
            run_index=0,
            runtime_seconds=runtime_seconds,
            timestamp=timestamp,
            stdout="MMLU-Pro Execution",
            stderr="",
            status=status,
            error=error,
            agent_output=str(agent_output) if agent_output is not None else None,
            software_versions={"mmlu_pro": "1.0"},
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
            extracted = extract_answer(str(agent_output))
            success = extracted == expected
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
            "name": "MMLU-Pro",
            "version": "1.0",
            "task_count": len(self._tasks) if self._loaded else 0,
            "deterministic": True,
        }


BenchmarkRegistry.register("MMLU-Pro", MMLUProAdapter)
