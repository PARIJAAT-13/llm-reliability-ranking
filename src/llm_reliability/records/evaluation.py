"""
Purpose
-------
Represent benchmark evaluation outcomes derived from a single execution.

Responsibilities
----------------
- Link evaluation results to their source ExecutionRecord via content hash
- Propagate execution identity and reliability-context fields without duplication
- Store success flag, normalized score, and benchmark-specific metrics
- Enforce derivation exclusively from ExecutionRecord data

Usage example
-------------
>>> from llm_reliability.records import EvaluationRecord, ExecutionRecord
>>> execution = ExecutionRecord(...)  # doctest: +SKIP
>>> evaluation = EvaluationRecord.from_execution(
...     execution,
...     success=True,
...     score=1.0,
...     evaluated_at="2026-01-01T00:00:00+00:00",
... )

Design notes
------------
EvaluationRecord never embeds the full ExecutionRecord to avoid duplication.
Instead it stores ``execution_hash`` as a reproducible foreign key alongside
identity fields copied verbatim from the execution. Factory method
``from_execution`` is the supported construction path from live execution
data; direct construction remains available only for canonical deserialization.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import Field

from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.utils.serialization import SerializableModel

EvaluationRecordT = TypeVar("EvaluationRecordT", bound="EvaluationRecord")


class EvaluationRecord(SerializableModel):
    """Immutable evaluation outcome derived from one ExecutionRecord."""

    execution_hash: str = Field(min_length=64, max_length=64)
    configuration_hash: str = Field(min_length=64, max_length=64)
    seed: int = Field(ge=0)
    benchmark: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    run_index: int = Field(ge=0)
    perturbation: str | None = None
    fault_injected: bool = False
    success: bool
    score: float = Field(ge=0.0, le=1.0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    evaluated_at: str = Field(min_length=1)

    @classmethod
    def from_execution(
        cls: type[EvaluationRecordT],
        execution: ExecutionRecord,
        *,
        success: bool,
        score: float,
        metrics: dict[str, Any] | None = None,
        evaluated_at: str,
    ) -> EvaluationRecordT:
        """Create an EvaluationRecord exclusively from an ExecutionRecord."""
        return cls(
            execution_hash=execution.sha256(),
            configuration_hash=execution.configuration_hash,
            seed=execution.seed,
            benchmark=execution.benchmark,
            agent=execution.agent,
            task_id=execution.task_id,
            run_index=execution.run_index,
            perturbation=execution.perturbation,
            fault_injected=execution.fault_injected,
            success=success,
            score=score,
            metrics=metrics or {},
            evaluated_at=evaluated_at,
        )
