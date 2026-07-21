"""
Purpose
-------
Define the abstract PerturbationStrategy contract and result containers for prompt
perturbation robustness evaluation.

Responsibilities
----------------
- Provide the abstract base class PerturbationStrategy that receives a task dict and
  returns a modified task dict while preserving semantic meaning.
- Provide PerturbationRunResult container for capturing baseline and perturbed execution
  and evaluation records.

Design notes
------------
Perturbation strategies operate on benchmark task payloads without modifying the original task dict.
Leaf execution records set their ``perturbation`` field to identify which strategy generated the variant,
enabling downstream computation of prompt perturbation robustness metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field

from llm_reliability.configs.config import Configuration
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.utils.serialization import SerializableModel


class PerturbationStrategy(ABC):
    """Abstract interface for prompt perturbation strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier/name of this perturbation strategy."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of what this strategy modifies."""

    @abstractmethod
    def apply(self, task: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
        """Apply perturbation to a task payload.

        Parameters
        ----------
        task : dict[str, Any]
            Original task dictionary (e.g. containing 'task_id', 'prompt', etc.).
        seed : int | None
            Optional random seed for deterministic variations.

        Returns
        -------
        dict[str, Any]
            A new task dictionary with perturbed prompt/content while preserving
            task identity and semantic meaning.
        """


class PerturbationRunResult(SerializableModel):
    """Immutable result container for baseline and perturbed task executions."""

    configuration: Configuration
    task_id: str = Field(min_length=1)
    original_task: dict[str, Any] = Field(default_factory=dict)
    perturbed_tasks: list[dict[str, Any]] = Field(default_factory=list)
    execution_records: list[ExecutionRecord] = Field(default_factory=list)
    evaluation_records: list[EvaluationRecord] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def baseline_execution(self) -> ExecutionRecord | None:
        """Return baseline execution record if present."""
        for rec in self.execution_records:
            if rec.perturbation is None:
                return rec
        return None

    @property
    def perturbed_executions(self) -> list[ExecutionRecord]:
        """Return list of perturbed execution records."""
        return [rec for rec in self.execution_records if rec.perturbation is not None]

    @property
    def baseline_evaluation(self) -> EvaluationRecord | None:
        """Return baseline evaluation record if present."""
        for rec in self.evaluation_records:
            if rec.perturbation is None:
                return rec
        return None

    @property
    def perturbed_evaluations(self) -> list[EvaluationRecord]:
        """Return list of perturbed evaluation records."""
        return [rec for rec in self.evaluation_records if rec.perturbation is not None]
