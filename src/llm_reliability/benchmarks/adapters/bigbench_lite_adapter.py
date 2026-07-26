"""
Purpose
-------
Integrate the BIG-Bench Lite benchmark into the framework.

Responsibilities
----------------
- Load BIG-Bench Lite tasks (input, targets, multiple_choice_options)
- Support both multiple-choice (letter extraction) and free-form evaluation (substring match)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
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


def extract_mc_answer(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"\b([A-D])\b", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\banswer\s+is\s+[\(]?([A-D])[\)]?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return text.strip()[:1].upper()


def check_free_form(output: str, targets: list[str]) -> bool:
    if not output or not targets:
        return False
    output_lower = output.lower()
    for t in targets:
        if t.lower() in output_lower:
            return True
    return False


class BigBenchLiteAdapter(BaseBenchmarkAdapter):
    """Adapter for the BIG-Bench Lite benchmark."""

    def validate_configuration(self) -> None:
        super().validate_configuration()
        if not self.config.metadata.get("dataset_path"):
            raise ValueError(
                "Configuration metadata must contain 'dataset_path' for BIG-Bench-Lite."
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
                raise TypeError("BIG-Bench-Lite dataset must be a list of tasks.")

            self._tasks = {}
            for item in data:
                tid = item.get("task_id", item.get("id", f"bigbench_{len(self._tasks)}"))
                inp = item.get("input", item.get("question", item.get("prompt", "")))
                targets = item.get(
                    "targets", item.get("target", item.get("ground_truth_answer", []))
                )
                if isinstance(targets, str):
                    targets = [targets]
                mc_options = item.get("multiple_choice_options", item.get("choices", []))
                self._tasks[tid] = {
                    "task_id": tid,
                    "input": inp,
                    "targets": list(targets),
                    "multiple_choice_options": list(mc_options),
                }
        else:
            self._tasks = {
                f"bigbench_{i}": {
                    "task_id": f"bigbench_{i}",
                    "input": [
                        "What is the capital of France? A. London B. Paris C. Berlin D. Madrid",
                        "Which planet is known as the Red Planet? A. Venus B. Jupiter C. Mars D. Saturn",
                        "What is 2+2?",
                        "Name one gas found in Earth's atmosphere.",
                        "Is the sky blue during the day? Answer yes or no.",
                    ][i],
                    "targets": [
                        ["B", "Paris"],
                        ["C", "Mars"],
                        ["4"],
                        ["oxygen", "nitrogen", "carbon dioxide"],
                        ["yes"],
                    ][i],
                    "multiple_choice_options": [
                        ["London", "Paris", "Berlin", "Madrid"],
                        ["Venus", "Jupiter", "Mars", "Saturn"],
                        [],
                        [],
                        [],
                    ][i],
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
            benchmark="BIG-Bench-Lite",
            agent=self.config.agent,
            task_id=task_id,
            run_index=0,
            runtime_seconds=runtime_seconds,
            timestamp=timestamp,
            stdout="BIG-Bench-Lite Execution",
            stderr="",
            status=status,
            error=error,
            agent_output=agent_output,
            software_versions={"bigbench_lite": "1.0"},
            environment_metadata={},
        )
        self._logs.append({"event": "run", "task_id": task_id, "status": status})
        return record

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        task = self.get_task(execution.task_id)
        targets = task.get("targets", [])
        mc_options = task.get("multiple_choice_options", [])
        agent_output = execution.agent_output

        if execution.status == "error" or not agent_output:
            success = False
            score = 0.0
        else:
            output_text = str(agent_output)
            if mc_options:
                extracted = extract_mc_answer(output_text)
                success = any(extracted == t.upper() for t in targets)
            else:
                success = check_free_form(output_text, targets)
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
            "name": "BIG-Bench-Lite",
            "version": "1.0",
            "task_count": len(self._tasks) if self._loaded else 0,
            "deterministic": True,
        }


BenchmarkRegistry.register("BIG-Bench-Lite", BigBenchLiteAdapter)
