"""
Purpose
-------
Compute repeated-run consistency metrics for LLM agents.

Responsibilities
----------------
- Compute success rate across repetitions.
- Compute response agreement rate across identical task executions.
- Compute score execution variance and latency variance.
- Produce deterministic consistency score.

Design notes
------------
Handles missing data gracefully by issuing logger warnings and returning zero variance/score
bounds when execution count is insufficient.
"""

from __future__ import annotations

import logging

import numpy as np

from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.reliability.metrics.base import (
    ConsistencyMetricResult,
    ReliabilityMetric,
)

logger = logging.getLogger(__name__)


class RepeatedRunConsistencyMetric(ReliabilityMetric):
    """Calculates repeated-run consistency metrics."""

    @property
    def name(self) -> str:
        return "repeated_run_consistency"

    @property
    def dimension(self) -> str:
        return "consistency"

    def compute(
        self,
        execution_records: list[ExecutionRecord],
        evaluation_records: list[EvaluationRecord],
    ) -> ConsistencyMetricResult:
        if not evaluation_records:
            logger.warning("No evaluation records provided for consistency metric computation.")
            return ConsistencyMetricResult(
                success_rate=0.0,
                response_agreement_rate=0.0,
                execution_variance=0.0,
                latency_variance=0.0,
                deterministic_consistency_score=0.0,
                metadata={"warning": "empty_evaluations"},
            )

        logger.info("Computing repeated-run consistency metrics across %d evaluations.", len(evaluation_records))

        # 1. Success rate
        scores = [ev.score for ev in evaluation_records]
        success_rate = float(np.mean(scores))

        # 2. Execution score variance
        execution_variance = float(np.var(scores)) if len(scores) > 1 else 0.0

        # 3. Latency variance
        runtimes = [rec.runtime_seconds for rec in execution_records]
        latency_variance = float(np.var(runtimes)) if len(runtimes) > 1 else 0.0

        # 4. Response agreement rate & deterministic consistency score
        task_groups: dict[str, list[ExecutionRecord]] = {}
        for rec in execution_records:
            task_groups.setdefault(rec.task_id, []).append(rec)

        task_agreements: list[float] = []
        task_majority_agreements: list[float] = []

        for task_id, recs in task_groups.items():
            if not recs:
                continue
            outputs = [str(r.agent_output) if r.agent_output is not None else "" for r in recs]
            # Output agreement (most frequent output count / total task runs)
            counts: dict[str, int] = {}
            for out in outputs:
                counts[out] = counts.get(out, 0) + 1
            max_count = max(counts.values()) if counts else 0
            task_agreements.append(max_count / len(outputs))

            # Majority success outcome agreement
            task_evals = [ev for ev in evaluation_records if ev.task_id == task_id]
            if task_evals:
                succ_vals = [ev.success for ev in task_evals]
                maj_succ = sum(succ_vals) >= len(succ_vals) / 2
                matching_maj = sum(s == maj_succ for s in succ_vals)
                task_majority_agreements.append(matching_maj / len(succ_vals))

        response_agreement_rate = (
            float(np.mean(task_agreements)) if task_agreements else 1.0
        )
        majority_agreement = (
            float(np.mean(task_majority_agreements)) if task_majority_agreements else 1.0
        )

        deterministic_consistency_score = float(
            np.clip(0.5 * response_agreement_rate + 0.5 * majority_agreement, 0.0, 1.0)
        )

        logger.info("Consistency computation complete: score=%.4f.", deterministic_consistency_score)

        return ConsistencyMetricResult(
            success_rate=success_rate,
            response_agreement_rate=response_agreement_rate,
            execution_variance=execution_variance,
            latency_variance=latency_variance,
            deterministic_consistency_score=deterministic_consistency_score,
            metadata={
                "n_executions": len(execution_records),
                "n_evaluations": len(evaluation_records),
                "n_tasks": len(task_groups),
            },
        )
