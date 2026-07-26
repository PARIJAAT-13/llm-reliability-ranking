"""
Purpose
-------
Compute prompt perturbation robustness metrics for LLM agents.

Responsibilities
----------------
- Compute success retention rate under prompt perturbations.
- Compute response stability across task variants.
- Compute perturbation sensitivity.
- Produce overall robustness score.

Design notes
------------
Handles missing perturbation data gracefully by issuing logger warnings and returning zero
sensitivity / standard metrics without raising exceptions.
"""

from __future__ import annotations

import logging

import numpy as np

from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.reliability.metrics.base import (ReliabilityMetric,
                                                      RobustnessMetricResult)

logger = logging.getLogger(__name__)


class PromptPerturbationRobustnessMetric(ReliabilityMetric):
    """Calculates prompt perturbation robustness metrics."""

    @property
    def name(self) -> str:
        return "prompt_perturbation_robustness"

    @property
    def dimension(self) -> str:
        return "robustness"

    def compute(
        self,
        execution_records: list[ExecutionRecord],
        evaluation_records: list[EvaluationRecord],
    ) -> RobustnessMetricResult:
        baseline_evals = [ev for ev in evaluation_records if ev.perturbation is None]
        perturbed_evals = [ev for ev in evaluation_records if ev.perturbation is not None]

        if not perturbed_evals:
            logger.warning(
                "No perturbation evaluation records found for robustness metric computation."
            )
            return RobustnessMetricResult(
                success_retention_rate=0.0,
                response_stability=0.0,
                perturbation_sensitivity=0.0,
                robustness_score=0.0,
                metadata={"warning": "no_perturbation_records"},
            )

        logger.info(
            "Computing robustness metrics across %d baseline and %d perturbed evaluations.",
            len(baseline_evals),
            len(perturbed_evals),
        )

        baseline_sr = float(np.mean([ev.score for ev in baseline_evals])) if baseline_evals else 0.0
        perturbed_sr = float(np.mean([ev.score for ev in perturbed_evals]))

        if baseline_sr == 0.0:
            success_retention_rate = 0.0 if perturbed_sr == 0.0 else 1.0
        else:
            success_retention_rate = float(np.clip(perturbed_sr / baseline_sr, 0.0, 1.0))

        perturbation_sensitivity = float(np.clip(1.0 - success_retention_rate, 0.0, 1.0))

        # Compute response stability across baseline vs perturbed executions per task
        baseline_execs = {rec.task_id: rec for rec in execution_records if rec.perturbation is None}
        perturbed_execs = [rec for rec in execution_records if rec.perturbation is not None]

        stability_matches: list[float] = []
        for p_exec in perturbed_execs:
            b_exec = baseline_execs.get(p_exec.task_id)
            if b_exec is not None:
                b_out = str(b_exec.agent_output) if b_exec.agent_output is not None else ""
                p_out = str(p_exec.agent_output) if p_exec.agent_output is not None else ""
                match_val = 1.0 if b_out.strip() == p_out.strip() else 0.0
                stability_matches.append(match_val)

        response_stability = (
            float(np.mean(stability_matches)) if stability_matches else success_retention_rate
        )

        robustness_score = float(
            np.clip(0.6 * success_retention_rate + 0.4 * response_stability, 0.0, 1.0)
        )

        logger.info("Robustness computation complete: score=%.4f.", robustness_score)

        return RobustnessMetricResult(
            success_retention_rate=success_retention_rate,
            response_stability=response_stability,
            perturbation_sensitivity=perturbation_sensitivity,
            robustness_score=robustness_score,
            metadata={
                "n_baseline": len(baseline_evals),
                "n_perturbed": len(perturbed_evals),
                "baseline_success_rate": baseline_sr,
                "perturbed_success_rate": perturbed_sr,
            },
        )
