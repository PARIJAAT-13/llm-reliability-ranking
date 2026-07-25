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

from __future__ import annotations

from llm_reliability.metrics.composite import compute_composite
from llm_reliability.metrics.consistency import compute_consistency
from llm_reliability.metrics.fault_tolerance import compute_fault_tolerance
from llm_reliability.metrics.isr import compute_isr, compute_temporal_isr
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

    def compute_isr(
        self,
        n_bins: int = 10,
        alpha: float = 0.6,
        ci_method: str | None = None,
        n_resamples: int = 1000,
        ci_alpha: float = 0.05,
        random_seed: int | None = None,
    ) -> dict | None:
        """Compute Information Survival Rate (ISR).

        Returns ``None`` if no fault-injected evaluations are present.

        Parameters
        ----------
        n_bins:
            Number of histogram bins for output-level ISR (default 10).
        alpha:
            Weight of output-level ISR in composite (default 0.6).
        ci_method:
            Confidence interval method (``"bootstrap"`` or ``None``).
        n_resamples:
            Bootstrap resamples (ignored unless ci_method=\"bootstrap\").
        ci_alpha:
            CI significance level (ignored unless ci_method=\"bootstrap\").
        random_seed:
            Seed for reproducible bootstrap.

        Returns
        -------
        dict | None
            ISR result dict or None.
        """
        has_faulted = any(ev.fault_injected for ev in self._evaluations)
        if not has_faulted:
            return None
        return compute_isr(
            self._evaluations,
            n_bins=n_bins,
            alpha=alpha,
            ci_method=ci_method,
            n_resamples=n_resamples,
            ci_alpha=ci_alpha,
            random_seed=random_seed,
        )

    def compute_temporal_isr(
        self,
        n_bins: int = 10,
        alpha: float = 0.6,
        n_windows: int = 5,
    ) -> dict | None:
        """Compute temporal (sequential) ISR over time windows.

        Returns ``None`` if no fault-injected evaluations are present.
        """
        has_faulted = any(ev.fault_injected for ev in self._evaluations)
        if not has_faulted:
            return None
        return compute_temporal_isr(
            self._evaluations,
            n_bins=n_bins,
            alpha=alpha,
            n_windows=n_windows,
        )

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
        isr_result = self.compute_isr()
        isr_comp = isr_result.get("isr_composite") if isr_result else None
        return compute_composite(
            success_rate=sr,
            consistency=cons,
            robustness=rob,
            fault_tolerance=ft,
            isr_composite=isr_comp,
            weights=weights,
        )

    # ------------------------------------------------------------------
    # All metrics in one call
    # ------------------------------------------------------------------

    def compute_all(
        self,
        task_id: str | None = None,
        weights: dict[str, float] | None = None,
        isr_ci_method: str | None = None,
        temporal_windows: int | None = None,
        isr_random_seed: int | None = None,
    ) -> ReliabilityResult:
        """Compute every metric and return a ReliabilityResult.

        Parameters
        ----------
        task_id:
            Optional task scope label to embed in the result.
        weights:
            Optional custom composite weights.
        isr_ci_method:
            Confidence interval method for ISR (``"bootstrap"`` or ``None``).
        temporal_windows:
            Number of temporal windows for temporal ISR.  ``None`` skips.
        isr_random_seed:
            Seed for reproducible bootstrap ISR.

        Returns
        -------
        ReliabilityResult
            Fully populated reliability result.
        """
        sr = self.compute_success_rate()
        cons = self.compute_consistency()
        rob = self.compute_robustness()
        ft = self.compute_fault_tolerance()
        isr_result = self.compute_isr(
            ci_method=isr_ci_method,
            random_seed=isr_random_seed,
        )

        isr_comp = isr_result.get("isr_composite") if isr_result else None
        composite, effective_weights = compute_composite(
            success_rate=sr,
            consistency=cons,
            robustness=rob,
            fault_tolerance=ft,
            isr_composite=isr_comp,
            weights=weights,
        )

        # Temporal ISR
        temporal_isr = None
        temporal_slope = None
        if temporal_windows is not None and isr_result is not None:
            t_result = self.compute_temporal_isr(n_windows=temporal_windows)
            if t_result:
                temporal_isr = t_result["window_isr"]
                temporal_slope = t_result["trend_slope"]

        return ReliabilityResult(
            task_id=task_id,
            success_rate=sr,
            consistency=cons,
            robustness=rob,
            fault_tolerance=ft,
            isr_output=isr_result.get("isr_output") if isr_result else None,
            isr_behavior=isr_result.get("isr_behavior") if isr_result else None,
            isr_composite=isr_result.get("isr_composite") if isr_result else None,
            isr_output_ci=isr_result.get("isr_output_ci") if isr_result else None,
            isr_behavior_ci=isr_result.get("isr_behavior_ci") if isr_result else None,
            temporal_isr=temporal_isr,
            temporal_isr_slope=temporal_slope,
            per_fault_type_isr=isr_result.get("per_fault_type", {}) if isr_result else {},
            composite=composite,
            weights=effective_weights,
            n_evaluations=len(self._evaluations),
        )
