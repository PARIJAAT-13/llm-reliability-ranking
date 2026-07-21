"""
Reliability Metrics Engine.

Orchestrates computation of all reliability metrics from a collection of
EvaluationRecords.  This is the single entry point for the metrics layer.

The engine does NOT execute benchmarks, rank agents, or produce final
rankings.  It only computes metrics and returns a ReliabilityResult.

Typical usage::

    from llm_reliability.metrics import ReliabilityEngine

    engine = ReliabilityEngine(evaluations)
    result = engine.compute_all()
"""

import numpy as np

from llm_reliability.metrics.composite import compute_composite
from llm_reliability.metrics.consistency import compute_consistency
from llm_reliability.metrics.fault_tolerance import compute_fault_tolerance
from llm_reliability.metrics.models import ReliabilityResult
from llm_reliability.metrics.robustness import compute_robustness
from llm_reliability.records.evaluation import EvaluationRecord


class ReliabilityEngine:
    """Compute all reliability metrics from a set of EvaluationRecords.

    Parameters
    ----------
    evaluations:
        The full list of EvaluationRecords to analyse.  Must not be empty.

    Raises
    ------
    ValueError
        If *evaluations* is empty.
    """

    def __init__(self, evaluations: list[EvaluationRecord]) -> None:
        if not evaluations:
            raise ValueError("ReliabilityEngine requires at least one EvaluationRecord.")
        self._evaluations = evaluations

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def compute_success_rate(self) -> float:
        """Compute success rate: successful_runs / total_runs.

        Returns
        -------
        float
            A value in [0, 1].
        """
        total = len(self._evaluations)
        successes = sum(1 for ev in self._evaluations if ev.success)
        return float(successes / total)

    def compute_consistency(self) -> float:
        """Compute repeated-run consistency.

        See :mod:`llm_reliability.metrics.consistency` for the full formula.

        Returns
        -------
        float
            A value in [0, 1].
        """
        return compute_consistency(self._evaluations)

    def compute_robustness(self) -> float | None:
        """Compute perturbation robustness.

        Returns ``None`` if no perturbed evaluations are present.

        Returns
        -------
        float | None
            A value in [0, 1] or None.
        """
        has_perturbed = any(ev.perturbation is not None for ev in self._evaluations)
        if not has_perturbed:
            return None
        return compute_robustness(self._evaluations)

    def compute_fault_tolerance(self) -> float | None:
        """Compute fault tolerance.

        Returns ``None`` if no fault-injected evaluations are present.

        Returns
        -------
        float | None
            A value in [0, 1] or None.
        """
        has_faulted = any(ev.fault_injected for ev in self._evaluations)
        if not has_faulted:
            return None
        return compute_fault_tolerance(self._evaluations)

    def compute_composite(
        self,
        weights: dict[str, float] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute the composite reliability score.

        Parameters
        ----------
        weights:
            Optional custom weight mapping.  If None, equal weights are used
            across all available (non-None) metrics.

        Returns
        -------
        tuple[float, dict[str, float]]
            (composite_score, effective_weights_used)
        """
        sr = self.compute_success_rate()
        cons = self.compute_consistency()
        rob = self.compute_robustness()
        ft = self.compute_fault_tolerance()
        return compute_composite(
            success_rate=sr,
            consistency=cons,
            robustness=rob,
            fault_tolerance=ft,
            weights=weights,
        )

    # ------------------------------------------------------------------
    # All metrics in one call
    # ------------------------------------------------------------------

    def compute_all(
        self,
        task_id: str | None = None,
        weights: dict[str, float] | None = None,
    ) -> ReliabilityResult:
        """Compute every metric and return a ReliabilityResult.

        Parameters
        ----------
        task_id:
            Optional task scope label to embed in the result.
        weights:
            Optional custom composite weights.

        Returns
        -------
        ReliabilityResult
            Fully populated reliability result.
        """
        sr = self.compute_success_rate()
        cons = self.compute_consistency()
        rob = self.compute_robustness()
        ft = self.compute_fault_tolerance()
        composite, effective_weights = compute_composite(
            success_rate=sr,
            consistency=cons,
            robustness=rob,
            fault_tolerance=ft,
            weights=weights,
        )

        return ReliabilityResult(
            task_id=task_id,
            success_rate=sr,
            consistency=cons,
            robustness=rob,
            fault_tolerance=ft,
            composite=composite,
            weights=effective_weights,
            n_evaluations=len(self._evaluations),
        )
