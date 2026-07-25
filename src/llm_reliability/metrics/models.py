"""
Pydantic models for all reliability metric outputs.

ReliabilityResult is the canonical output of the Reliability Metrics Engine.
It carries every computed metric alongside provenance information so the
upstream ranking layer can consume a single, self-contained object.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from llm_reliability.utils.serialization import SerializableModel


class ReliabilityResult(SerializableModel):
    """Immutable container for all reliability metrics computed from a set
    of EvaluationRecords.

    Fields
    ------
    task_id
        The task identifier these metrics were computed for, or ``None`` when
        computed over all tasks in a run.
    success_rate
        Fraction of runs that succeeded: ``successful / total``.  Range [0, 1].
    consistency
        Agreement across repeated executions of the same task.  Range [0, 1].
    robustness
        Performance retention under prompt perturbations.  Range [0, 1] or
        ``None`` when no perturbation records are present.
    fault_tolerance
        Performance retention after injected failures.  Range [0, 1] or
        ``None`` when no fault-injection records are present.
    composite
        Weighted average of the above scores.  Range [0, 1].
    weights
        The weight vector used when computing ``composite``.
    n_evaluations
        Total number of EvaluationRecords used.
    metadata
        Arbitrary provenance / diagnostic information.
    """

    task_id: str | None = None
    success_rate: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)
    robustness: float | None = Field(default=None, ge=0.0, le=1.0)
    fault_tolerance: float | None = Field(default=None, ge=0.0, le=1.0)
    isr_output: float | None = Field(default=None, ge=0.0, le=1.0)
    isr_behavior: float | None = Field(default=None, ge=0.0, le=1.0)
    isr_composite: float | None = Field(default=None, ge=0.0, le=1.0)
    isr_output_ci: tuple[float, float] | None = Field(default=None)
    isr_behavior_ci: tuple[float, float] | None = Field(default=None)
    temporal_isr: list[float] | None = Field(default=None)
    temporal_isr_slope: float | None = Field(default=None)
    per_fault_type_isr: dict[str, float] = Field(default_factory=dict)
    composite: float = Field(ge=0.0, le=1.0)
    weights: dict[str, float] = Field(default_factory=dict)
    n_evaluations: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> ReliabilityResult:
        if self.weights:
            total = sum(self.weights.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Metric weights must sum to 1.0, got {total:.6f}.")
        return self
