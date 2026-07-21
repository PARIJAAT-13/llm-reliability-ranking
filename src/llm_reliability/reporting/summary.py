"""
ExperimentSummary — portable context object for all report generators.

Purpose
-------
Aggregate all experiment artifacts into a single validated Pydantic model
that every report generator consumes.  This decouples report rendering from
record loading and ensures consistent data across Markdown, LaTeX, and HTML
outputs.

Responsibilities
----------------
- Collect MetricRecords, RankingRecords, StatisticalReport
- Store experiment metadata and timestamp
- Provide convenience properties for common derived views

Usage example
-------------
>>> from llm_reliability.reporting.summary import ExperimentSummary
>>> summary = ExperimentSummary(
...     experiment_id="exp-001",
...     experiment_name="Pilot Study",
...     metrics=metrics,
...     rankings=rankings,
...     statistical_report=report,
... )
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ExperimentSummary(BaseModel):
    """Aggregated experiment context passed to all report generators.

    Attributes
    ----------
    experiment_id : str
        Unique experiment identifier.
    experiment_name : str
        Human-readable name.
    generated_at : str
        ISO-8601 UTC timestamp when the summary was created.
    metrics : list
        All MetricRecord objects.
    rankings : list
        All RankingRecord objects.
    executions : list
        All ExecutionRecord objects (may be empty for summary-only reports).
    evaluations : list
        All EvaluationRecord objects (may be empty).
    statistical_report : Any | None
        A StatisticalReport from the Statistical Analysis Engine, or None.
    config_snapshot : dict
        Arbitrary configuration key-value pairs for reproducibility.
    metadata : dict
        Additional experiment metadata (tags, notes, etc.).
    """

    model_config = {"arbitrary_types_allowed": True}

    experiment_id: str = Field(min_length=1)
    experiment_name: str = Field(min_length=1)
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metrics: list[Any] = Field(default_factory=list)
    rankings: list[Any] = Field(default_factory=list)
    executions: list[Any] = Field(default_factory=list)
    evaluations: list[Any] = Field(default_factory=list)
    statistical_report: Any | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def benchmarks(self) -> list[str]:
        """Unique benchmark names present in the metrics."""
        return sorted({m.benchmark for m in self.metrics})

    @property
    def agents(self) -> list[str]:
        """Unique agent names present in the metrics."""
        return sorted({m.agent for m in self.metrics})

    @property
    def success_rankings(self) -> list[Any]:
        """All RankingRecords with ranking_type == 'success'."""
        return [r for r in self.rankings if getattr(r, "ranking_type", "") == "success"]

    @property
    def reliability_rankings(self) -> list[Any]:
        """All RankingRecords with ranking_type == 'reliability'."""
        return [r for r in self.rankings if getattr(r, "ranking_type", "") == "reliability"]

    @property
    def weighted_rankings(self) -> list[Any]:
        """All RankingRecords with ranking_type == 'weighted'."""
        return [r for r in self.rankings if getattr(r, "ranking_type", "") == "weighted"]

    @property
    def n_evaluations(self) -> int:
        """Total number of evaluations across all metric records."""
        return sum(getattr(m, "evaluation_count", 0) for m in self.metrics)

    @property
    def n_executions(self) -> int:
        """Number of execution records."""
        return len(self.executions)

    def metrics_for_benchmark(self, benchmark: str) -> list[Any]:
        """Return MetricRecords for a specific benchmark."""
        return [m for m in self.metrics if m.benchmark == benchmark]

    def rankings_for_benchmark(self, benchmark: str) -> list[Any]:
        """Return RankingRecords for a specific benchmark."""
        return [r for r in self.rankings if getattr(r, "benchmark", "") == benchmark]

    @classmethod
    def from_experiment_runner(
        cls,
        runner: Any,
        experiment_name: str,
        statistical_report: Any | None = None,
        config_snapshot: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ExperimentSummary":
        """Build from a completed ``ExperimentRunner`` instance.

        Parameters
        ----------
        runner : ExperimentRunner
            A runner whose ``run()`` method has been called.
        experiment_name : str
            Experiment display name.
        statistical_report : StatisticalReport | None
            Pre-computed statistical analysis.
        config_snapshot : dict | None
            Configuration key-value pairs.
        metadata : dict | None
            Additional metadata.

        Returns
        -------
        ExperimentSummary
        """
        spec = getattr(runner, "_spec", None)
        exp_id = getattr(spec, "experiment_id", "unknown")

        return cls(
            experiment_id=exp_id,
            experiment_name=experiment_name,
            metrics=list(runner.metrics),
            rankings=list(runner.rankings),
            executions=list(runner.executions),
            evaluations=list(runner.evaluations),
            statistical_report=statistical_report,
            config_snapshot=config_snapshot or {},
            metadata=metadata or {},
        )
