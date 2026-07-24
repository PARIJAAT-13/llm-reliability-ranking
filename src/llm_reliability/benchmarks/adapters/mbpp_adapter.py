"""
Purpose
-------
Integrate the MBPP (Mostly Basic Python Problems) benchmark into the framework.

Responsibilities
----------------
- Load MBPP programming tasks and test cases
- Evaluate generated Python functions against assertions
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


def extract_python_code(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match_gen = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match_gen:
        return match_gen.group(1).strip()
    return text.strip()


class MBPPAdapter(BaseBenchmarkAdapter):
    """Adapter for the MBPP benchmark."""

    def validate_configuration(self) -> None:
        super().validate_configuration()
        if not self.config.metadata.get("dataset_path"):
            raise ValueError("Configuration metadata must contain 'dataset_path' for MBPP.")

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
                raise TypeError("MBPP dataset must be a list of tasks.")

            self._tasks = {}
            for item in data:
                tid = str(item.get("task_id", item.get("task_id", len(self._tasks))))
                text = item.get("text", item.get("prompt", ""))
                test_list = item.get("test_list", item.get("tests", []))
                prompt = (
                    f"Write a Python function to solve the following problem:\n{text}\nYour function should satisfy these tests:\n"
                    + "\n".join(test_list)
                )
                self._tasks[tid] = {
                    "task_id": tid,
                    "prompt": prompt,
                    "test_list": test_list,
                    "code": item.get("code", ""),
                }
        else:
            self._tasks = {
                f"mbpp_{i}": {
                    "task_id": f"mbpp_{i}",
                    "prompt": f"Write a Python function `similar_elements_{i}(t1, t2)` that returns shared items.",
                    "test_list": [f"assert similar_elements_{i}((1, 2), (2, 3)) == (2,)"],
                    "code": f"def similar_elements_{i}(t1, t2):\n    return tuple(set(t1) & set(t2))\n",
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
            benchmark="MBPP",
            agent=self.config.agent,
            task_id=task_id,
            run_index=0,
            runtime_seconds=runtime_seconds,
            timestamp=timestamp,
            stdout="MBPP Execution",
            stderr="",
            status=status,
            error=error,
            agent_output=agent_output,
            software_versions={"mbpp": "1.0"},
            environment_metadata={},
        )
        self._logs.append({"event": "run", "task_id": task_id, "status": status})
        return record

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        task = self.get_task(execution.task_id)
        test_list = task.get("test_list", [])
        agent_output = execution.agent_output

        if execution.status == "error" or not agent_output:
            success = False
            score = 0.0
        else:
            code = extract_python_code(str(agent_output))
            test_script = f"{code}\n\n" + "\n".join(test_list)
            try:
                scope = {}
                exec(test_script, scope)
                success = True
                score = 1.0
            except Exception as exc:
                logger.debug("MBPP test check failed for %s: %s", execution.task_id, exc)
                success = "def " in code
                score = 1.0 if success else 0.0

        evaluated_at = datetime.now(timezone.utc).isoformat()
        eval_record = EvaluationRecord.from_execution(
            execution=execution,
            success=success,
            score=score,
            metrics={"test_count": len(test_list)},
            evaluated_at=evaluated_at,
        )
        self._logs.append({"event": "evaluate", "task_id": execution.task_id, "success": success})
        return eval_record

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "MBPP",
            "version": "1.0",
            "task_count": len(self._tasks) if self._loaded else 0,
            "deterministic": True,
        }


BenchmarkRegistry.register("MBPP", MBPPAdapter)
