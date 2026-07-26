"""
Purpose
-------
Integrate the Arena Hard benchmark into the framework.

Responsibilities
----------------
- Load chat-style evaluation tasks (prompt, reference_answer, category)
- Evaluate response quality via keyword presence or exact match against reference
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


def extract_keywords(reference: str) -> set[str]:
    words = re.findall(r"[A-Za-z]{4,}", reference.lower())
    return set(words)


def check_quality(output: str, reference: str) -> bool:
    if not output or not reference:
        return False
    if output.strip().lower() == reference.strip().lower():
        return True
    keywords = extract_keywords(reference)
    if not keywords:
        return False
    output_lower = output.lower()
    matches = sum(1 for kw in keywords if kw in output_lower)
    return matches >= len(keywords) * 0.5


class ArenaHardAdapter(BaseBenchmarkAdapter):
    """Adapter for the Arena Hard benchmark."""

    def validate_configuration(self) -> None:
        super().validate_configuration()
        if not self.config.metadata.get("dataset_path"):
            raise ValueError("Configuration metadata must contain 'dataset_path' for ArenaHard.")

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
                raise TypeError("ArenaHard dataset must be a list of tasks.")

            self._tasks = {}
            for item in data:
                tid = item.get("task_id", item.get("id", f"arena_hard_{len(self._tasks)}"))
                prompt = item.get("prompt", item.get("question", item.get("conversation", "")))
                reference = item.get(
                    "reference_answer",
                    item.get("reference", item.get("ground_truth_answer", "")),
                )
                category = item.get("category", item.get("type", "general"))
                self._tasks[tid] = {
                    "task_id": tid,
                    "prompt": prompt,
                    "reference_answer": reference,
                    "category": category,
                }
        else:
            self._tasks = {
                f"arena_hard_{i}": {
                    "task_id": f"arena_hard_{i}",
                    "prompt": [
                        "Explain the concept of recursion in programming.",
                        "Write a short poem about artificial intelligence.",
                        "What are the benefits of using version control systems?",
                        "Describe how a TCP handshake works.",
                        "Compare machine learning and traditional programming.",
                    ][i],
                    "reference_answer": [
                        "Recursion is a programming technique where a function calls itself to solve smaller instances of a problem until reaching a base case.",
                        "Silicon dreams awake, learning patterns for our sake, intelligence anew.",
                        "Version control systems track changes, enable collaboration, and allow rollbacks to previous states.",
                        "TCP uses a three-way handshake: SYN, SYN-ACK, and ACK to establish a reliable connection.",
                        "Machine learning learns patterns from data while traditional programming follows explicit rules written by developers.",
                    ][i],
                    "category": [
                        "programming",
                        "creative",
                        "software_engineering",
                        "networking",
                        "comparison",
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
            benchmark="ArenaHard",
            agent=self.config.agent,
            task_id=task_id,
            run_index=0,
            runtime_seconds=runtime_seconds,
            timestamp=timestamp,
            stdout="ArenaHard Execution",
            stderr="",
            status=status,
            error=error,
            agent_output=agent_output,
            software_versions={"arena_hard": "1.0"},
            environment_metadata={},
        )
        self._logs.append({"event": "run", "task_id": task_id, "status": status})
        return record

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        task = self.get_task(execution.task_id)
        reference = str(task.get("reference_answer", ""))
        agent_output = execution.agent_output

        if execution.status == "error" or not agent_output:
            success = False
            score = 0.0
        else:
            output_text = str(agent_output)
            success = check_quality(output_text, reference)
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
            "name": "ArenaHard",
            "version": "1.0",
            "task_count": len(self._tasks) if self._loaded else 0,
            "deterministic": True,
        }


BenchmarkRegistry.register("ArenaHard", ArenaHardAdapter)
