"""
Purpose
-------
Integrate the HotpotQA (multi-hop QA) benchmark into the framework.

Responsibilities
----------------
- Load HotpotQA multi-hop reasoning tasks
- Evaluate via exact match and F1 score
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_reliability.benchmarks.adapters.base_adapter import BaseBenchmarkAdapter
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry
from llm_reliability.interfaces.agent import Agent
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_exact_match(response: str, answer: str) -> bool:
    return normalize_text(response) == normalize_text(answer)


def compute_f1(response: str, answer: str) -> float:
    norm_response = normalize_text(response)
    norm_answer = normalize_text(answer)
    response_tokens = norm_response.split()
    answer_tokens = norm_answer.split()

    if not answer_tokens:
        return 0.0
    if not response_tokens:
        return 0.0

    common = Counter(response_tokens) & Counter(answer_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(response_tokens)
    recall = num_common / len(answer_tokens)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


class HotpotQAAdapter(BaseBenchmarkAdapter):
    """Adapter for the HotpotQA benchmark."""

    def validate_configuration(self) -> None:
        super().validate_configuration()
        if not self.config.metadata.get("dataset_path"):
            raise ValueError("Configuration metadata must contain 'dataset_path' for HotpotQA.")

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
                raise TypeError("HotpotQA dataset must be a list of tasks.")

            self._tasks = {}
            for item in data:
                tid = item.get("task_id", f"hotpotqa_{len(self._tasks)}")
                question = item.get("question", item.get("prompt", ""))
                context = item.get("context", item.get("supporting_facts", []))
                answer = item.get("answer", item.get("ground_truth_answer", ""))

                context_text = ""
                if isinstance(context, list):
                    if context and isinstance(context[0], list):
                        parts = []
                        for fact in context:
                            if isinstance(fact, list) and len(fact) >= 2:
                                parts.append(
                                    f"{fact[0]}: {' '.join(fact[1]) if isinstance(fact[1], list) else str(fact[1])}"
                                )
                            else:
                                parts.append(str(fact))
                        context_text = "\n".join(parts)
                    else:
                        context_text = "\n".join(str(c) for c in context)
                elif isinstance(context, str):
                    context_text = context

                prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"

                self._tasks[tid] = {
                    "task_id": tid,
                    "question": question,
                    "context": context,
                    "prompt": prompt,
                    "ground_truth_answer": str(answer),
                }
        else:
            self._tasks = {
                f"hotpotqa_{i}": {
                    "task_id": f"hotpotqa_{i}",
                    "question": f"What is the capital of the country where the Eiffel Tower is located? {i}",
                    "context": [
                        [
                            "France",
                            ["Paris is the capital of France.", "The Eiffel Tower is in Paris."],
                        ]
                    ],
                    "prompt": f"Context:\nFrance: Paris is the capital of France. The Eiffel Tower is in Paris.\n\nQuestion: What is the capital of the country where the Eiffel Tower is located? {i}\nAnswer:",
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
            benchmark="HotpotQA",
            agent=self.config.agent,
            task_id=task_id,
            run_index=0,
            runtime_seconds=runtime_seconds,
            timestamp=timestamp,
            stdout="HotpotQA Execution",
            stderr="",
            status=status,
            error=error,
            agent_output=agent_output,
            software_versions={"hotpotqa": "1.0"},
            environment_metadata={},
        )
        self._logs.append({"event": "run", "task_id": task_id, "status": status})
        return record

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        task = self.get_task(execution.task_id)
        expected = str(task.get("ground_truth_answer", ""))
        agent_output = execution.agent_output

        if execution.status == "error" or not agent_output:
            success = False
            score = 0.0
            em = 0.0
            f1 = 0.0
        else:
            response_str = str(agent_output)
            em = 1.0 if compute_exact_match(response_str, expected) else 0.0
            f1 = compute_f1(response_str, expected)
            score = f1
            success = em > 0

        evaluated_at = datetime.now(timezone.utc).isoformat()
        eval_record = EvaluationRecord.from_execution(
            execution=execution,
            success=success,
            score=score,
            metrics={"exact_match": em, "f1": f1},
            evaluated_at=evaluated_at,
        )
        self._logs.append({"event": "evaluate", "task_id": execution.task_id, "success": success})
        return eval_record

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "HotpotQA",
            "version": "1.0",
            "task_count": len(self._tasks) if self._loaded else 0,
            "deterministic": True,
        }


BenchmarkRegistry.register("HotpotQA", HotpotQAAdapter)
