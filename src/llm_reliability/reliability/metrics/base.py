"""
Purpose
-------
Define the abstract ReliabilityMetric interface and structured metric result models
for the quantitative evaluation of agent reliability.

Responsibilities
----------------
- Provide abstract base class ReliabilityMetric.
- Provide structured Pydantic models for ConsistencyMetricResult, RobustnessMetricResult,
  and FaultToleranceMetricResult.

Design notes
------------
Metric calculation classes consume ExecutionRecord and EvaluationRecord objects and
return structured Pydantic model outputs (never plain dicts), preserving provenance
metadata and compatibility across pipeline components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field

from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.utils.serialization import SerializableModel


class ConsistencyMetricResult(SerializableModel):
    """Structured result model for repeated-run consistency metrics."""

    success_rate: float = Field(ge=0.0, le=1.0)
    response_agreement_rate: float = Field(ge=0.0, le=1.0)
    execution_variance: float = Field(ge=0.0)
    latency_variance: float = Field(ge=0.0)
    deterministic_consistency_score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RobustnessMetricResult(SerializableModel):
    """Structured result model for prompt perturbation robustness metrics."""

    success_retention_rate: float = Field(ge=0.0, le=1.0)
    response_stability: float = Field(ge=0.0, le=1.0)
    perturbation_sensitivity: float = Field(ge=0.0, le=1.0)
    robustness_score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FaultToleranceMetricResult(SerializableModel):
    """Structured result model for fault tolerance metrics."""

    recovery_rate: float = Field(ge=0.0, le=1.0)
    retry_success_rate: float = Field(ge=0.0, le=1.0)
    failure_resilience: float = Field(ge=0.0, le=1.0)
    avg_recovery_latency_seconds: float = Field(ge=0.0)
    fault_tolerance_score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReliabilityMetric(ABC):
    """Abstract interface for individual reliability metric calculators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""

    @property
    @abstractmethod
    def dimension(self) -> str:
        """Return the reliability dimension name (e.g. 'consistency', 'robustness', 'fault_tolerance')."""

    @abstractmethod
    def compute(
        self,
        execution_records: list[ExecutionRecord],
        evaluation_records: list[EvaluationRecord],
    ) -> SerializableModel:
        """Compute the metric from execution and evaluation records.

        Parameters
        ----------
        execution_records : list[ExecutionRecord]
            Execution audit records.
        evaluation_records : list[EvaluationRecord]
            Evaluation outcome records.

        Returns
        -------
        SerializableModel
            Structured metric result object.
        """
