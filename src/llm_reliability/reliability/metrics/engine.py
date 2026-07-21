"""
Purpose
-------
Coordinate quantitative reliability metric calculation across repeated runs, prompt perturbations,
and fault injections.

Responsibilities
----------------
- Group execution and evaluation records per agent, per benchmark, and overall scope.
- Execute metric calculators for consistency, robustness, and fault tolerance dimensions.
- Produce structured per-agent, per-benchmark, and overall metric results.
- Export standard MetricRecord objects compatible with the framework ranking pipeline.

Design notes
------------
ReliabilityMetricsEngine logs warnings for missing execution scopes without raising errors.
It connects leaf execution/evaluation records to upstream MetricRecords and ranking algorithms.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord
from llm_reliability.reliability.metrics.base import (
    ConsistencyMetricResult,
    FaultToleranceMetricResult,
    ReliabilityMetric,
    RobustnessMetricResult,
)
from llm_reliability.reliability.metrics.consistency import (
    RepeatedRunConsistencyMetric,
)
from llm_reliability.reliability.metrics.fault_tolerance import FaultToleranceMetric
from llm_reliability.reliability.metrics.robustness import (
    PromptPerturbationRobustnessMetric,
)

logger = logging.getLogger(__name__)


class ScopeReliabilitySummary:
    """Summary metrics container for a single scope (agent, benchmark, or overall)."""

    def __init__(
        self,
        scope_name: str,
        consistency: ConsistencyMetricResult,
        robustness: RobustnessMetricResult,
        fault_tolerance: FaultToleranceMetricResult,
        composite_score: float,
        metric_record: MetricRecord | None = None,
    ) -> None:
        self.scope_name = scope_name
        self.consistency = consistency
        self.robustness = robustness
        self.fault_tolerance = fault_tolerance
        self.composite_score = composite_score
        self.metric_record = metric_record

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_name": self.scope_name,
            "consistency": self.consistency.model_dump(),
            "robustness": self.robustness.model_dump(),
            "fault_tolerance": self.fault_tolerance.model_dump(),
            "composite_score": self.composite_score,
            "metric_record": self.metric_record.model_dump() if self.metric_record else None,
        }


class ReliabilityMetricsEngine:
    """Coordinates calculation of quantitative reliability metrics across dimensions."""

    def __init__(
        self,
        metrics: list[ReliabilityMetric] | None = None,
    ) -> None:
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = [
                RepeatedRunConsistencyMetric(),
                PromptPerturbationRobustnessMetric(),
                FaultToleranceMetric(),
            ]

    def compute_all(
        self,
        execution_records: list[ExecutionRecord],
        evaluation_records: list[EvaluationRecord],
    ) -> dict[str, Any]:
        """Compute reliability metrics grouped per agent, per benchmark, and overall.

        Parameters
        ----------
        execution_records : list[ExecutionRecord]
            Execution audit records.
        evaluation_records : list[EvaluationRecord]
            Evaluation outcome records.

        Returns
        -------
        dict[str, Any]
            Dictionary containing 'per_agent', 'per_benchmark', 'overall', and 'metric_records'.
        """
        logger.info(
            "ReliabilityMetricsEngine starting computation: %d executions, %d evaluations.",
            len(execution_records),
            len(evaluation_records),
        )

        if not evaluation_records:
            logger.warning("ReliabilityMetricsEngine received empty evaluation_records.")

        # Group by Agent
        agent_groups: dict[str, tuple[list[ExecutionRecord], list[EvaluationRecord]]] = {}
        for ev in evaluation_records:
            agent_name = ev.agent
            if agent_name not in agent_groups:
                execs = [rec for rec in execution_records if rec.agent == agent_name]
                evals = [rec for rec in evaluation_records if rec.agent == agent_name]
                agent_groups[agent_name] = (execs, evals)

        # Group by Benchmark
        bench_groups: dict[str, tuple[list[ExecutionRecord], list[EvaluationRecord]]] = {}
        for ev in evaluation_records:
            bench_name = ev.benchmark
            if bench_name not in bench_groups:
                execs = [rec for rec in execution_records if rec.benchmark == bench_name]
                evals = [rec for rec in evaluation_records if rec.benchmark == bench_name]
                bench_groups[bench_name] = (execs, evals)

        per_agent: dict[str, ScopeReliabilitySummary] = {}
        metric_records: list[MetricRecord] = []

        for agent_name, (execs, evals) in agent_groups.items():
            summary = self._compute_scope(f"agent:{agent_name}", execs, evals)
            per_agent[agent_name] = summary
            if summary.metric_record:
                metric_records.append(summary.metric_record)

        per_benchmark: dict[str, ScopeReliabilitySummary] = {}
        for bench_name, (execs, evals) in bench_groups.items():
            summary = self._compute_scope(f"benchmark:{bench_name}", execs, evals)
            per_benchmark[bench_name] = summary

        overall = self._compute_scope("overall", execution_records, evaluation_records)

        logger.info("ReliabilityMetricsEngine calculation completed successfully.")

        return {
            "per_agent": per_agent,
            "per_benchmark": per_benchmark,
            "overall": overall,
            "metric_records": metric_records,
        }

    def _compute_scope(
        self,
        scope_name: str,
        execution_records: list[ExecutionRecord],
        evaluation_records: list[EvaluationRecord],
    ) -> ScopeReliabilitySummary:
        """Compute metrics for a specific grouping scope."""
        consistency_res: ConsistencyMetricResult | None = None
        robustness_res: RobustnessMetricResult | None = None
        fault_res: FaultToleranceMetricResult | None = None

        for metric in self.metrics:
            try:
                res = metric.compute(execution_records, evaluation_records)
                if isinstance(res, ConsistencyMetricResult):
                    consistency_res = res
                elif isinstance(res, RobustnessMetricResult):
                    robustness_res = res
                elif isinstance(res, FaultToleranceMetricResult):
                    fault_res = res
            except Exception as exc:
                logger.error("Error computing metric '%s' for scope '%s': %s", metric.name, scope_name, exc, exc_info=True)

        if consistency_res is None:
            consistency_res = ConsistencyMetricResult(
                success_rate=0.0,
                response_agreement_rate=0.0,
                execution_variance=0.0,
                latency_variance=0.0,
                deterministic_consistency_score=0.0,
            )

        if robustness_res is None:
            robustness_res = RobustnessMetricResult(
                success_retention_rate=0.0,
                response_stability=0.0,
                perturbation_sensitivity=0.0,
                robustness_score=0.0,
            )

        if fault_res is None:
            fault_res = FaultToleranceMetricResult(
                recovery_rate=0.0,
                retry_success_rate=0.0,
                failure_resilience=0.0,
                avg_recovery_latency_seconds=0.0,
                fault_tolerance_score=0.0,
            )

        # Composite score calculation
        scores = [consistency_res.deterministic_consistency_score]
        if "warning" not in robustness_res.metadata:
            scores.append(robustness_res.robustness_score)
        if "warning" not in fault_res.metadata:
            scores.append(fault_res.fault_tolerance_score)

        composite_score = sum(scores) / len(scores) if scores else 0.0

        # Build framework MetricRecord if evaluations exist
        metric_record: MetricRecord | None = None
        if evaluation_records:
            try:
                now_str = datetime.now(timezone.utc).isoformat()
                metric_record = MetricRecord.from_evaluations(
                    evaluation_records,
                    computed_at=now_str,
                )
            except Exception as exc:
                logger.warning("Failed to create MetricRecord for scope '%s': %s", scope_name, exc)

        return ScopeReliabilitySummary(
            scope_name=scope_name,
            consistency=consistency_res,
            robustness=robustness_res,
            fault_tolerance=fault_res,
            composite_score=composite_score,
            metric_record=metric_record,
        )
