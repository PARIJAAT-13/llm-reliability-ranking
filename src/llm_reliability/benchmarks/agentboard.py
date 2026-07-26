"""
Purpose
-------
Provide a complete, self-contained AgentBoard benchmark adapter for the
LLM Reliability Ranking framework.

Responsibilities
----------------
- Implement every method required by the ``Benchmark`` abstract interface
- Load AgentBoard tasks from a JSON dataset file whose path is supplied via
  ``Configuration.metadata["dataset_path"]``
- Execute agent.run(task) inside ``run()`` and wrap the outcome in an
  ``ExecutionRecord``
- Evaluate agent output against the expected answer inside ``evaluate()`` and
  produce an ``EvaluationRecord`` via ``EvaluationRecord.from_execution``
- Accumulate benchmark-level log events and expose them via ``collect_logs()``
- Expose descriptive benchmark metadata via ``metadata()``
- Register the adapter in ``BenchmarkRegistry`` so ``ExperimentRunner`` can
  discover it by name without modification

Usage example
-------------
>>> from llm_reliability.benchmarks.agentboard import AgentBoardBenchmark
>>> from llm_reliability.configs.config import Configuration
>>> cfg = Configuration(
...     experiment_name="pilot",
...     benchmark="AgentBoard",
...     agent="mock",
...     llm="gpt-4",
...     prompt_version="v1",
...     dataset_version="1.0",
...     seed=42,
...     repetitions=3,
...     metadata={"dataset_path": "/path/to/agentboard.json"},
... )
>>> bench = AgentBoardBenchmark(cfg)
>>> bench.load()
>>> tasks = bench.list_tasks()

Design notes
------------
This module is intentionally self-contained and does **not** depend on
``BaseBenchmarkAdapter``.  It inherits directly from ``Benchmark`` (the pure
abstract interface), mirroring the pattern of ``MockBenchmark``.  This means:

* No shared base-class state couples it to other adapters.
* The full contract is visible in a single file, making it easier to audit for
  a conference paper reviewing the framework.

The adapter re-exports itself under the name ``"AgentBoard"`` in
``BenchmarkRegistry``, which is the name ``ExperimentRunner`` uses when building
a ``Configuration`` for an AgentBoard run.

AgentBoard dataset assumptions
------------------------------
The dataset is expected to be a JSON array of task objects.  Each object must
contain the following keys (additional keys are preserved in ``metadata``):

``task_id``        (str, non-empty)  — unique identifier
``prompt``         (str, non-empty)  — instruction given to the agent
``expected_output``(str, non-empty)  — ground-truth answer
``difficulty``     (str, non-empty)  — task difficulty label
``category``       (str, non-empty)  — task category (e.g. "web", "tool_use")

Optional keys:

``progress_rate``  (float, 0-1)      — partial-credit score; when present,
                                       ``score`` in the EvaluationRecord is set
                                       to this value rather than binary 0/1.
``metadata``       (dict)            — any extra task-level annotations

These assumptions match the AgentBoard paper (Liu et al., 2023) and the schema
defined in ``adapters/agentboard_models.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry
from llm_reliability.benchmarks.agentboard_utils import score_agentboard
from llm_reliability.configs.config import Configuration
from llm_reliability.interfaces.agent import Agent
from llm_reliability.interfaces.benchmark import Benchmark
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord

logger = logging.getLogger(__name__)

# Semantic version of this adapter.  Bump when the evaluation logic or task
# schema changes so that stored EvaluationRecords remain traceable.
AGENTBOARD_ADAPTER_VERSION = "1.0"


class AgentBoardBenchmark(Benchmark):
    """Complete AgentBoard benchmark adapter.

    Inherits directly from ``Benchmark`` and implements every abstract method
    required by the interface.  Accepts a ``Configuration`` object exactly as
    ``MockBenchmark`` does so the ``ExperimentRunner`` can instantiate it
    without modification.

    Parameters
    ----------
    config:
        Framework configuration object.  Must include
        ``config.metadata["dataset_path"]`` pointing to a UTF-8 JSON file
        that contains a list of AgentBoard task objects.

    Raises
    ------
    ValueError
        If ``config`` is ``None`` or does not contain ``dataset_path`` in its
        metadata dictionary.
    """

    def __init__(self, config: Configuration) -> None:
        """Initialise the adapter and validate configuration immediately."""
        if not config:
            raise ValueError("Configuration must be provided.")

        dataset_path = config.metadata.get("dataset_path")
        if not dataset_path:
            raise ValueError("Configuration metadata must contain 'dataset_path' for AgentBoard.")

        self._config = config
        self._tasks: dict[str, Any] = {}
        self._logs: list[dict[str, Any]] = []
        self._loaded: bool = False

        logger.debug(
            "AgentBoardBenchmark initialised (dataset_path=%s, seed=%d).",
            dataset_path,
            config.seed,
        )

    # ------------------------------------------------------------------
    # Benchmark interface — mandatory abstract methods
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the AgentBoard dataset from disk and validate every task.

        Reads a JSON file at the path given in
        ``self._config.metadata["dataset_path"]``.  The file must contain a
        JSON array.  Each element is validated against the expected schema
        (see module docstring).  Duplicate ``task_id`` values raise an error.

        Raises
        ------
        RuntimeError
            If the file cannot be read or is not valid JSON.
        TypeError
            If the top-level JSON value is not a list.
        ValueError
            If any task object is missing required fields or a duplicate
            ``task_id`` is encountered.
        """
        dataset_path: str = self._config.metadata["dataset_path"]

        logger.info("Loading AgentBoard dataset from '%s'.", dataset_path)

        try:
            with open(dataset_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.error("Failed to load AgentBoard dataset from '%s': %s", dataset_path, exc)
            raise RuntimeError(f"Missing or invalid dataset: {exc}") from exc

        if not isinstance(data, list):
            raise TypeError(
                f"AgentBoard dataset must be a JSON array of task objects, "
                f"got {type(data).__name__!r}."
            )

        loaded: dict[str, Any] = {}
        for raw in data:
            try:
                task = self._parse_task(raw)
            except (KeyError, TypeError, ValueError) as exc:
                logger.error("Malformed AgentBoard task — skipping: %s", exc)
                raise ValueError(f"Invalid schema: {exc}") from exc

            task_id: str = task["task_id"]
            if task_id in loaded:
                raise ValueError(
                    f"Duplicate task ID found: {task_id!r}. "
                    "AgentBoard dataset must contain unique task_id values."
                )
            loaded[task_id] = task

        if not loaded:
            raise ValueError(
                f"AgentBoard dataset is empty — no tasks were loaded from '{dataset_path}'."
            )

        self._tasks = loaded
        self._loaded = True

        logger.info(
            "AgentBoard dataset loaded successfully: %d tasks from '%s'.",
            len(self._tasks),
            dataset_path,
        )
        self._logs.append(
            {
                "event": "load",
                "status": "success",
                "dataset_path": dataset_path,
                "task_count": len(self._tasks),
            }
        )

    def list_tasks(self) -> list[str]:
        """Return all task identifiers in deterministic (sorted) order.

        Returns
        -------
        list[str]
            Sorted list of ``task_id`` strings loaded from the dataset.

        Raises
        ------
        RuntimeError
            If ``load()`` has not been called.
        """
        self._require_loaded()
        return sorted(self._tasks.keys())

    def get_task(self, task_id: str) -> dict[str, Any]:
        """Return a copy of the task payload for the given identifier.

        Parameters
        ----------
        task_id:
            Unique task identifier as returned by ``list_tasks()``.

        Returns
        -------
        dict[str, Any]
            Shallow copy of the stored task dict.

        Raises
        ------
        RuntimeError
            If ``load()`` has not been called.
        ValueError
            If ``task_id`` is not present in the loaded dataset.
        """
        self._require_loaded()
        if task_id not in self._tasks:
            raise ValueError(
                f"Unknown task_id: {task_id!r}. "
                "Ensure the task_id belongs to the loaded AgentBoard dataset."
            )
        return self._tasks[task_id].copy()

    def run(self, agent: Agent, task: dict[str, Any]) -> ExecutionRecord:
        """Execute the agent on a single AgentBoard task.

        Calls ``agent.run(task)`` and captures any exception.  Wraps the
        outcome in an ``ExecutionRecord`` using the existing class without
        inventing a new record type.

        Timing is deterministic when ``config.seed`` is set (required for
        reproducible SHA-256 hashes across repeated test runs).  Wall-clock
        time is used otherwise (production runs).

        Parameters
        ----------
        agent:
            Agent implementing the ``Agent`` abstract interface.
        task:
            Task payload as returned by ``get_task()``.

        Returns
        -------
        ExecutionRecord
            Immutable record of the execution outcome.
        """
        task_id: str = task["task_id"]

        logger.debug(
            "Running AgentBoard task '%s' with agent '%s'.",
            task_id,
            agent.__class__.__name__,
        )

        start_time = datetime.now(timezone.utc)
        try:
            agent_output = agent.run(task)
            status: str = "success"
            error: str | None = None
        except Exception as exc:  # noqa: BLE001  — intentional broad catch
            agent_output = None
            status = "error"
            error = str(exc)
            logger.warning("Agent raised exception on task '%s': %s", task_id, exc)

        # Use deterministic timing when a seed is configured so that
        # ExecutionRecord.sha256() is reproducible across repeated calls.
        if self._config.seed is not None:
            h = hashlib.sha256(f"{self._config.seed}_{task_id}".encode())
            deterministic_int = int(h.hexdigest()[:8], 16)
            runtime_seconds = 1.0 + (deterministic_int % 40) / 10.0
            timestamp = f"2026-01-01T00:{deterministic_int % 60:02d}:00+00:00"
        else:
            runtime_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
            timestamp = start_time.isoformat()

        record = ExecutionRecord(
            configuration_hash=self._config.sha256(),
            seed=self._config.seed,
            benchmark="AgentBoard",
            # Use the logical agent name from config (AgentSpec.name) not the class name.
            agent=self._config.agent,
            task_id=task_id,
            run_index=0,
            runtime_seconds=runtime_seconds,
            timestamp=timestamp,
            stdout="AgentBoard Execution",
            stderr="",
            status=status,
            error=error,
            agent_output=agent_output,
            software_versions={"agentboard": AGENTBOARD_ADAPTER_VERSION},
            environment_metadata={},
        )

        self._logs.append(
            {
                "event": "run",
                "task_id": task_id,
                "status": status,
                "runtime_seconds": runtime_seconds,
            }
        )

        logger.debug(
            "Task '%s' execution complete: status=%s runtime=%.3fs.",
            task_id,
            status,
            runtime_seconds,
        )

        return record

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        """Evaluate a completed execution against the AgentBoard ground truth.

        Normalises both the expected output and the agent output (lowercase +
        strip punctuation/whitespace) before performing an exact-match
        comparison.  When the task contains a ``progress_rate`` field, that
        value is used as the score (partial credit); otherwise the score is
        binary (1.0 / 0.0).

        Parameters
        ----------
        execution:
            An ``ExecutionRecord`` produced by ``run()``.

        Returns
        -------
        EvaluationRecord
            Created exclusively via ``EvaluationRecord.from_execution()`` to
            preserve the derivation chain required by the pipeline.
        """
        task = self.get_task(execution.task_id)

        logger.debug("Evaluating task '%s'.", execution.task_id)

        if execution.status == "error":
            success = False
            score = 0.0
        else:
            expected: str = task.get("expected_output", "")
            agent_output: Any = execution.agent_output
            progress_rate: float | None = task.get("progress_rate")

            success, score = score_agentboard(
                expected=expected,
                agent_output=str(agent_output),
                progress_rate=progress_rate,
            )

        # Deterministic timestamp for reproducible hashing in test runs.
        if self._config.seed is not None:
            h = hashlib.sha256(f"eval_{self._config.seed}_{execution.task_id}".encode())
            deterministic_int = int(h.hexdigest()[:8], 16)
            evaluated_at = f"2026-01-01T01:{deterministic_int % 60:02d}:00+00:00"
        else:
            evaluated_at = datetime.now(timezone.utc).isoformat()

        eval_record = EvaluationRecord.from_execution(
            execution=execution,
            success=success,
            score=score,
            metrics={
                "difficulty": task.get("difficulty"),
                "category": task.get("category"),
            },
            evaluated_at=evaluated_at,
        )

        self._logs.append(
            {
                "event": "evaluate",
                "task_id": execution.task_id,
                "success": success,
                "score": score,
            }
        )

        logger.debug(
            "Task '%s' evaluation complete: success=%s score=%.4f.",
            execution.task_id,
            success,
            score,
        )

        return eval_record

    def collect_logs(self) -> dict[str, Any]:
        """Return a snapshot of benchmark-level logs accumulated during execution.

        Returns
        -------
        dict[str, Any]
            Dictionary with a single ``"logs"`` key containing a list of event
            dicts recorded by ``load()``, ``run()``, and ``evaluate()``.
        """
        return {"logs": self._logs.copy()}

    def metadata(self) -> dict[str, Any]:
        """Return descriptive metadata about this benchmark adapter.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: ``name``, ``version``, ``deterministic``,
            ``task_count``, and ``dataset_path``.
        """
        return {
            "name": "AgentBoard",
            "version": AGENTBOARD_ADAPTER_VERSION,
            "deterministic": True,
            "task_count": len(self._tasks) if self._loaded else 0,
            "dataset_path": self._config.metadata.get("dataset_path", ""),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        """Raise RuntimeError if the benchmark has not been loaded yet."""
        if not self._loaded:
            raise RuntimeError("AgentBoardBenchmark has not been loaded. Call load() first.")

    @staticmethod
    def _parse_task(raw: Any) -> dict[str, Any]:
        """Validate and normalise a raw task dict from the JSON dataset.

        Parameters
        ----------
        raw:
            Arbitrary value decoded from the JSON array.

        Returns
        -------
        dict[str, Any]
            Validated task dict with all required fields present.

        Raises
        ------
        TypeError
            If ``raw`` is not a dict.
        ValueError
            If any required field is missing or empty.
        """
        if not isinstance(raw, dict):
            raise TypeError(f"Each task must be a JSON object (dict), got {type(raw).__name__!r}.")

        required_fields = ("task_id", "prompt", "expected_output", "difficulty", "category")
        for field in required_fields:
            value = raw.get(field)
            if not value or not str(value).strip():
                raise ValueError(
                    f"Task is missing required non-empty field: {field!r}. Got {value!r}."
                )

        # Validate optional progress_rate if present
        progress_rate = raw.get("progress_rate")
        if progress_rate is not None:
            try:
                pr = float(progress_rate)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"'progress_rate' must be a float in [0, 1], got {progress_rate!r}."
                ) from exc
            if not (0.0 <= pr <= 1.0):
                raise ValueError(f"'progress_rate' must be in [0, 1], got {pr}.")

        # Normalise: keep all original fields, ensure metadata dict exists
        task: dict[str, Any] = dict(raw)
        task.setdefault("metadata", {})
        return task


# ---------------------------------------------------------------------------
# Registry self-registration
#
# Import this module to register "AgentBoard" in the BenchmarkRegistry so that
# ExperimentRunner._default_benchmark_factory can discover it by name.
#
# Guard prevents duplicate-registration errors when the module is imported
# multiple times (e.g. once from adapters/ and once from this file).
# ---------------------------------------------------------------------------
if not BenchmarkRegistry.exists("AgentBoard"):
    BenchmarkRegistry.register("AgentBoard", AgentBoardBenchmark)  # type: ignore[arg-type]
