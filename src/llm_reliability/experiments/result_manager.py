"""
Result Manager for the Experiment Runner.

Handles serialization, checkpointing, and structured on-disk persistence of
all experiment artifacts into the canonical output directory tree.

Output structure
----------------
results/
  <experiment_id>/
    configuration.json
    executions.json
    evaluations.json
    metrics.json
    rankings.json
    statistics.json
    metadata.json
    checkpoint.json     ← resumes from here on restart
    logs/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord
from llm_reliability.experiments.experiment_models import ExperimentSpec, ExperimentStatus

logger = logging.getLogger(__name__)


class ResultManager:
    """Manages on-disk persistence of experiment artifacts.

    Parameters
    ----------
    spec : ExperimentSpec
        The experiment specification.
    output_dir : str | Path
        Root directory for outputs (default: ``results/``).
    """

    def __init__(self, spec: ExperimentSpec, output_dir: str | Path = "results") -> None:
        self._spec = spec
        self._root = Path(output_dir) / spec.experiment_id
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / "logs").mkdir(exist_ok=True)

    @property
    def experiment_dir(self) -> Path:
        """Return the experiment output directory."""
        return self._root

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def _write_json(self, filename: str, data: Any) -> None:
        """Write data to a JSON file inside the experiment directory."""
        path = self._root / filename
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def save_checkpoint(self, completed_run_indices: list[int]) -> None:
        """Persist completed run indices so the experiment can be resumed.

        Parameters
        ----------
        completed_run_indices : list[int]
            Zero-based indices of fully completed RunDescriptors.
        """
        self._write_json("checkpoint.json", {"completed": completed_run_indices})
        logger.debug("Checkpoint saved: %d runs completed.", len(completed_run_indices))

    def load_checkpoint(self) -> list[int]:
        """Load completed run indices from disk.

        Returns
        -------
        list[int]
            Previously completed run indices, or an empty list if no checkpoint.
        """
        path = self._root / "checkpoint.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("completed", [])

    # ------------------------------------------------------------------
    # Artifact saves
    # ------------------------------------------------------------------

    def save_configuration(self) -> None:
        """Write ExperimentSpec to configuration.json."""
        self._write_json("configuration.json", self._spec.canonical_json())

    def save_executions(self, records: list[ExecutionRecord]) -> None:
        """Write ExecutionRecords to executions.json."""
        payload = [json.loads(r.canonical_json()) for r in records]
        self._write_json("executions.json", payload)

    def save_evaluations(self, records: list[EvaluationRecord]) -> None:
        """Write EvaluationRecords to evaluations.json."""
        payload = [json.loads(r.canonical_json()) for r in records]
        self._write_json("evaluations.json", payload)

    def save_metrics(self, records: list[MetricRecord]) -> None:
        """Write MetricRecords to metrics.json."""
        payload = [json.loads(r.canonical_json()) for r in records]
        self._write_json("metrics.json", payload)

    def save_rankings(self, records: list[RankingRecord]) -> None:
        """Write RankingRecords to rankings.json."""
        payload = [json.loads(r.canonical_json()) for r in records]
        self._write_json("rankings.json", payload)

    def save_statistics(self, report: Any) -> None:
        """Write the statistical report to statistics.json."""
        if hasattr(report, "model_dump_json"):
            self._write_json("statistics.json", json.loads(report.model_dump_json()))
        else:
            self._write_json("statistics.json", report)

    def save_metadata(self, status: ExperimentStatus, extra: dict[str, Any] | None = None) -> None:
        """Write experiment metadata to metadata.json."""
        payload: dict[str, Any] = {
            "experiment_id": self._spec.experiment_id,
            "experiment_name": self._spec.experiment_name,
            "state": status.state,
            "started_at": status.started_at,
            "completed_at": status.completed_at,
            "total_runs": status.total_runs,
            "completed_runs": status.completed_runs,
            "failed_runs": status.failed_runs,
            "errors": status.errors,
        }
        if extra:
            payload.update(extra)
        self._write_json("metadata.json", payload)

    def save_status(self, status: ExperimentStatus) -> None:
        """Save status JSON for progress polling."""
        self._write_json("status.json", json.loads(status.canonical_json()))

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def load_executions(self) -> list[dict[str, Any]]:
        """Load raw executions.json records."""
        path = self._root / "executions.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def load_evaluations(self) -> list[dict[str, Any]]:
        """Load raw evaluations.json records."""
        path = self._root / "evaluations.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))
