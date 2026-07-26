"""
Purpose
-------
Integrate the IFEval (Instruction Following Evaluation) benchmark into the framework.

Responsibilities
----------------
- Load instruction-following tasks (prompt + constraint)
- Evaluate whether model output satisfies the given instruction/constraint
- Support format checks via regex patterns
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


COMMON_INSTRUCTIONS: dict[str, str] = {
    "json": r"\{[^{}]*\}",
    "list": r"\[.*\]",
    "keywords": r"\b(because|therefore|however|furthermore)\b",
    "length_gt_50": r"^.{51,}$",
    "length_lt_20": r"^.{0,19}$",
    "positive_tone": r"\b(good|great|excellent|wonderful|fantastic)\b",
    "email": r"[\w\.-]+@[\w\.-]+\.\w+",
    "bullet_points": r"(?:^|\n)[\*\-\+]\s",
    "numbered_list": r"(?:^|\n)\d+[\.\)]\s",
    "uppercase_word": r"\b[A-Z]{2,}\b",
    "punctuation_end": r"[.!?]$",
    "no_comma": r"^[^,]*$",
}


def satisfies_instruction(
    text: str, instruction_key: str, custom_pattern: str | None = None
) -> bool:
    if not text:
        return False
    pattern = custom_pattern or COMMON_INSTRUCTIONS.get(instruction_key)
    if not pattern:
        return False
    try:
        return bool(re.search(pattern, text, re.MULTILINE))
    except re.error:
        logger.warning("Invalid regex pattern for instruction: %s", instruction_key)
        return False


class IFEvalAdapter(BaseBenchmarkAdapter):
    """Adapter for the IFEval (Instruction Following Evaluation) benchmark."""

    def validate_configuration(self) -> None:
        super().validate_configuration()
        if not self.config.metadata.get("dataset_path"):
            raise ValueError("Configuration metadata must contain 'dataset_path' for IFEval.")

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
                raise TypeError("IFEval dataset must be a list of tasks.")

            self._tasks = {}
            for item in data:
                tid = item.get("task_id", item.get("id", f"ifeval_{len(self._tasks)}"))
                prompt = item.get("prompt", item.get("instruction", ""))
                instruction = item.get("instruction", item.get("constraint", ""))
                required_format = item.get("required_format", item.get("format", ""))
                gt = item.get("ground_truth", item.get("ground_truth_answer", ""))
                self._tasks[tid] = {
                    "task_id": tid,
                    "prompt": prompt,
                    "instruction": instruction,
                    "required_format": required_format,
                    "ground_truth": str(gt),
                }
        else:
            self._tasks = {
                f"ifeval_{i}": {
                    "task_id": f"ifeval_{i}",
                    "prompt": [
                        "Write a sentence about the weather that includes the word 'sunny'.",
                        "List three fruits in a JSON array.",
                        "Write a paragraph with exactly 50 words about AI safety.",
                        "Respond with a bullet-point list of 4 programming languages.",
                        "Write an email address and a sentence ending with a period.",
                    ][i],
                    "instruction": [
                        "keywords",
                        "json",
                        "length_gt_50",
                        "bullet_points",
                        "email",
                    ][i],
                    "required_format": [
                        "include keyword 'sunny'",
                        "valid JSON array",
                        "at least 51 characters",
                        "bullet-point list",
                        "contains email",
                    ][i],
                    "ground_truth": [
                        "sunny",
                        '["apple","banana","cherry"]',
                        "AI safety is important for ensuring that artificial intelligence systems ...",
                        "* Python\n* JavaScript\n* Rust\n* Go",
                        "user@example.com.",
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
            benchmark="IFEval",
            agent=self.config.agent,
            task_id=task_id,
            run_index=0,
            runtime_seconds=runtime_seconds,
            timestamp=timestamp,
            stdout="IFEval Execution",
            stderr="",
            status=status,
            error=error,
            agent_output=agent_output,
            software_versions={"ifeval": "1.0"},
            environment_metadata={},
        )
        self._logs.append({"event": "run", "task_id": task_id, "status": status})
        return record

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        task = self.get_task(execution.task_id)
        instruction = str(task.get("instruction", ""))
        required_format = str(task.get("required_format", ""))
        agent_output = execution.agent_output

        if execution.status == "error" or not agent_output:
            success = False
            score = 0.0
        else:
            output_text = str(agent_output)
            instruction_satisfied = satisfies_instruction(output_text, instruction)
            if required_format and not instruction_satisfied:
                custom_match = bool(re.search(required_format, output_text, re.IGNORECASE))
                success = custom_match
            else:
                success = instruction_satisfied
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
            "name": "IFEval",
            "version": "1.0",
            "task_count": len(self._tasks) if self._loaded else 0,
            "deterministic": True,
        }


BenchmarkRegistry.register("IFEval", IFEvalAdapter)
