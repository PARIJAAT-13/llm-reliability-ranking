"""
Purpose
-------
Define the abstract FaultInjectionStrategy contract and result models for fault tolerance evaluation.

Responsibilities
----------------
- Provide the abstract FaultInjectionStrategy base class defining fault_name, injection_point,
  inject(), and cleanup() lifecycle methods.
- Provide FaultTrace to record execution telemetry during fault injection.
- Provide FaultRunResult container for baseline and fault-injected ExecutionRecord and EvaluationRecord artifacts.

Design notes
------------
Fault injection strategies inject controlled failures (timeouts, API failures, invalid responses,
tool failures, context truncation, network errors) into the execution environment.
Fault-injected executions set ``fault_injected = True`` on ExecutionRecord and EvaluationRecord,
enabling downstream computation of fault tolerance metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import Field

from llm_reliability.configs.config import Configuration
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.utils.serialization import SerializableModel

RecoveryStatus = Literal["success", "partial", "failed"]


class FaultInjectionStrategy(ABC):
    """Abstract interface for all fault injection strategies."""

    @property
    @abstractmethod
    def fault_name(self) -> str:
        """Return unique identifier name for this fault strategy."""

    @property
    @abstractmethod
    def injection_point(self) -> str:
        """Return the injection hook point (e.g. 'agent_run', 'api_call', 'prompt', 'tool_call')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Return human-readable description of the injected fault."""

    @abstractmethod
    def inject(self, target: Any, seed: int | None = None, **kwargs: Any) -> Any:
        """Inject fault into the execution target.

        Parameters
        ----------
        target : Any
            Target payload or object (e.g. task dict, prompt string, agent, API call).
        seed : int | None
            Random seed for reproducible fault generation.
        **kwargs : Any
            Additional context parameters.

        Returns
        -------
        Any
            Modified target or result after fault injection.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any active mocks, hooks, or state after fault injection."""


class FaultTrace(SerializableModel):
    """Record capturing details of an injected fault occurrence and agent recovery."""

    fault_name: str = Field(min_length=1)
    injection_point: str = Field(min_length=1)
    retry_count: int = Field(ge=0)
    recovery_status: RecoveryStatus
    execution_outcome: str = Field(min_length=1)
    latency_seconds: float = Field(ge=0.0)
    details: dict[str, Any] = Field(default_factory=dict)


class FaultRunResult(SerializableModel):
    """Immutable collection of all records produced during a fault tolerance experiment."""

    configuration: Configuration
    task_id: str = Field(min_length=1)
    original_task: dict[str, Any] = Field(default_factory=dict)
    execution_records: list[ExecutionRecord] = Field(default_factory=list)
    evaluation_records: list[EvaluationRecord] = Field(default_factory=list)
    fault_traces: list[FaultTrace] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def baseline_execution(self) -> ExecutionRecord | None:
        """Return non-faulted baseline execution record if present."""
        for rec in self.execution_records:
            if not rec.fault_injected:
                return rec
        return None

    @property
    def faulted_executions(self) -> list[ExecutionRecord]:
        """Return list of fault-injected execution records."""
        return [rec for rec in self.execution_records if rec.fault_injected]

    @property
    def baseline_evaluation(self) -> EvaluationRecord | None:
        """Return non-faulted baseline evaluation record if present."""
        for rec in self.evaluation_records:
            if not rec.fault_injected:
                return rec
        return None

    @property
    def faulted_evaluations(self) -> list[EvaluationRecord]:
        """Return list of fault-injected evaluation records."""
        return [rec for rec in self.evaluation_records if rec.fault_injected]
