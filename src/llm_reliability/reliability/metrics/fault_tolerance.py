"""
Purpose
-------
Compute fault tolerance metrics for LLM agents under injected failures.

Responsibilities
----------------
- Compute recovery rate.
- Compute retry success rate.
- Compute failure resilience.
- Compute average recovery latency.
- Produce overall fault tolerance score.

Design notes
------------
Handles missing fault injection records gracefully by issuing logger warnings and returning
structured zero scores with warning metadata without raising exceptions.
"""

from __future__ import annotations

import logging

import numpy as np

from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.reliability.metrics.base import (
    FaultToleranceMetricResult,
    ReliabilityMetric,
)

logger = logging.getLogger(__name__)


class FaultToleranceMetric(ReliabilityMetric):
    """Calculates fault tolerance metrics."""

    @property
    def name(self) -> str:
        return "fault_tolerance"

    @property
    def dimension(self) -> str:
        return "fault_tolerance"

    def compute(
        self,
        execution_records: list[ExecutionRecord],
        evaluation_records: list[EvaluationRecord],
    ) -> FaultToleranceMetricResult:
        baseline_evals = [ev for ev in evaluation_records if not ev.fault_injected]
        faulted_evals = [ev for ev in evaluation_records if ev.fault_injected]
        faulted_execs = [rec for rec in execution_records if rec.fault_injected]

        if not faulted_evals:
            logger.warning(
                "No fault injection evaluation records found for fault tolerance metric computation."
            )
            return FaultToleranceMetricResult(
                recovery_rate=0.0,
                retry_success_rate=0.0,
                failure_resilience=0.0,
                avg_recovery_latency_seconds=0.0,
                fault_tolerance_score=0.0,
                metadata={"warning": "no_fault_records"},
            )

        logger.info(
            "Computing fault tolerance metrics across %d baseline and %d faulted evaluations.",
            len(baseline_evals),
            len(faulted_evals),
        )

        baseline_sr = float(np.mean([ev.score for ev in baseline_evals])) if baseline_evals else 0.0
        faulted_sr = float(np.mean([ev.score for ev in faulted_evals]))

        # Recovery rate (evaluations with score > 0 or success=True)
        recovered_evals = [ev for ev in faulted_evals if ev.success or ev.score > 0.0]
        recovery_rate = len(recovered_evals) / len(faulted_evals) if faulted_evals else 0.0

        # Failure resilience
        if baseline_sr == 0.0:
            failure_resilience = 0.0 if faulted_sr == 0.0 else 1.0
        else:
            failure_resilience = float(np.clip(faulted_sr / baseline_sr, 0.0, 1.0))

        # Retry success rate & average recovery latency from telemetry
        recovery_latencies: list[float] = []
        retry_successes = 0
        total_retried_attempts = 0

        for f_exec in faulted_execs:
            telemetry = f_exec.environment_metadata.get("fault_injection", {})
            retries = telemetry.get("retry_count", 0)
            status = telemetry.get("recovery_status", "failed")

            if retries > 0:
                total_retried_attempts += 1
                if status in ("success", "partial") or f_exec.status == "success":
                    retry_successes += 1

            if status in ("success", "partial") or f_exec.status == "success":
                latency = telemetry.get("latency_seconds", f_exec.runtime_seconds)
                recovery_latencies.append(latency)

        retry_success_rate = (
            retry_successes / total_retried_attempts
            if total_retried_attempts > 0
            else recovery_rate
        )

        avg_recovery_latency = (
            float(np.mean(recovery_latencies))
            if recovery_latencies
            else float(np.mean([r.runtime_seconds for r in faulted_execs]))
        )

        fault_tolerance_score = float(
            np.clip(0.5 * recovery_rate + 0.5 * failure_resilience, 0.0, 1.0)
        )

        logger.info("Fault tolerance computation complete: score=%.4f.", fault_tolerance_score)

        return FaultToleranceMetricResult(
            recovery_rate=recovery_rate,
            retry_success_rate=retry_success_rate,
            failure_resilience=failure_resilience,
            avg_recovery_latency_seconds=avg_recovery_latency,
            fault_tolerance_score=fault_tolerance_score,
            metadata={
                "n_baseline": len(baseline_evals),
                "n_faulted": len(faulted_evals),
                "baseline_success_rate": baseline_sr,
                "faulted_success_rate": faulted_sr,
            },
        )
