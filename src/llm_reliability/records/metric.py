"""
Purpose
-------
Aggregate evaluation outcomes into reliability and success metrics per agent.

Responsibilities
----------------
- Compute success rate and reliability sub-metrics from EvaluationRecords
- Support task-level and benchmark-level aggregation scopes
- Produce composite reliability scores from component metrics

Usage example
-------------
>>> from llm_reliability.records import MetricRecord, EvaluationRecord
>>> metric = MetricRecord.from_evaluations([...], evaluated_at="2026-01-01T00:00:00+00:00")

Design notes
------------
MetricRecord derives exclusively from EvaluationRecord instances. Component
reliability metrics (consistency, perturbation robustness, fault tolerance)
may be ``None`` when insufficient evaluation data exists for computation.
The composite score averages only available components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic import Field, model_validator

from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.utils.serialization import SerializableModel

if TYPE_CHECKING:
    pass


def _compute_repeated_run_consistency(evaluations: list[EvaluationRecord]) -> float:
    """Fraction of evaluations with identical success outcomes within a run group."""
    if not evaluations:
        return 0.0
    success_values = [evaluation.success for evaluation in evaluations]
    majority = sum(success_values) >= len(success_values) / 2
    consistent = sum(value == majority for value in success_values)
    return consistent / len(success_values)


def _compute_perturbation_robustness(
    evaluations: list[EvaluationRecord],
) -> float | None:
    """Compare baseline vs perturbed success rates when perturbation data exists."""
    baseline = [evaluation for evaluation in evaluations if evaluation.run_index == 0]
    perturbed = [evaluation for evaluation in evaluations if evaluation.run_index > 0]
    if not baseline or not perturbed:
        return None
    baseline_rate = sum(item.success for item in baseline) / len(baseline)
    perturbed_rate = sum(item.success for item in perturbed) / len(perturbed)
    if baseline_rate == 0.0:
        return perturbed_rate
    return min(perturbed_rate / baseline_rate, 1.0)


def _compute_fault_tolerance(evaluations: list[EvaluationRecord]) -> float | None:
    """Success rate under fault-injection conditions when data is present."""
    fault_evaluations = [evaluation for evaluation in evaluations if evaluation.fault_injected]
    if not fault_evaluations:
        return None
    return sum(item.success for item in fault_evaluations) / len(fault_evaluations)


def _compute_composite(
    success_rate: float,
    consistency: float,
    perturbation: float | None,
    fault_tolerance: float | None,
    isr_composite: float | None = None,
    cost_efficiency: float | None = None,
) -> float:
    """Average available reliability components into a composite score."""
    components = [success_rate, consistency]
    if perturbation is not None:
        components.append(perturbation)
    if fault_tolerance is not None:
        components.append(fault_tolerance)
    if isr_composite is not None:
        components.append(isr_composite)
    if cost_efficiency is not None:
        components.append(cost_efficiency)
    return sum(components) / len(components)


MetricRecordT = TypeVar("MetricRecordT", bound="MetricRecord")


class MetricRecord(SerializableModel):
    """Immutable reliability and success metrics for one agent scope."""

    benchmark: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    task_id: str | None = None
    evaluation_count: int = Field(ge=1)
    success_rate: float = Field(ge=0.0, le=1.0)
    repeated_run_consistency: float = Field(ge=0.0, le=1.0)
    perturbation_robustness: float | None = Field(default=None, ge=0.0, le=1.0)
    fault_tolerance: float | None = Field(default=None, ge=0.0, le=1.0)
    isr_output: float | None = Field(default=None, ge=0.0, le=1.0)
    isr_behavior: float | None = Field(default=None, ge=0.0, le=1.0)
    isr_composite_val: float | None = Field(default=None, ge=0.0, le=1.0)
    total_cost_usd: float | None = Field(default=None, ge=0.0)
    cost_per_success: float | None = Field(default=None, ge=0.0)
    cost_efficiency_score: float | None = Field(default=None, ge=0.0, le=1.0)
    cost_weighted_reliability: float | None = Field(default=None, ge=0.0, le=1.0)
    composite_reliability: float = Field(ge=0.0, le=1.0)
    computed_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_composite_bounds(self) -> MetricRecord:
        """Ensure composite reliability stays within [0, 1]."""
        if not 0.0 <= self.composite_reliability <= 1.0:
            msg = "composite_reliability must be in [0.0, 1.0]"
            raise ValueError(msg)
        return self

    @classmethod
    def from_evaluations(
        cls: type[MetricRecordT],
        evaluations: list[EvaluationRecord],
        *,
        task_id: str | None = None,
        computed_at: str,
    ) -> MetricRecordT:
        """Derive metrics exclusively from a non-empty list of evaluations."""
        if not evaluations:
            msg = "evaluations must contain at least one EvaluationRecord"
            raise ValueError(msg)

        benchmark = evaluations[0].benchmark
        agent = evaluations[0].agent
        if any(item.benchmark != benchmark or item.agent != agent for item in evaluations):
            msg = "all evaluations must share the same benchmark and agent"
            raise ValueError(msg)

        success_rate = sum(item.success for item in evaluations) / len(evaluations)
        consistency = _compute_repeated_run_consistency(evaluations)
        perturbation = _compute_perturbation_robustness(evaluations)
        fault_tolerance = _compute_fault_tolerance(evaluations)

        has_faulted = any(ev.fault_injected for ev in evaluations)
        if has_faulted:
            from llm_reliability.reliability.metrics.isr import (
                compute_isr as _compute_isr,
            )

            isr_result = _compute_isr(evaluations)
            isr_out = isr_result["isr_output"]
            isr_beh = isr_result["isr_behavior"]
            isr_comp = isr_result["isr_composite"]
        else:
            isr_out = isr_beh = isr_comp = None

        composite = _compute_composite(
            success_rate, consistency, perturbation, fault_tolerance, isr_comp
        )

        return cls(
            benchmark=benchmark,
            agent=agent,
            task_id=task_id,
            evaluation_count=len(evaluations),
            success_rate=success_rate,
            repeated_run_consistency=consistency,
            perturbation_robustness=perturbation,
            fault_tolerance=fault_tolerance,
            isr_output=isr_out,
            isr_behavior=isr_beh,
            isr_composite_val=isr_comp,
            composite_reliability=composite,
            computed_at=computed_at,
        )
