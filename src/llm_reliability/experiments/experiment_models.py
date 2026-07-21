"""
Pydantic v2 models for the Experiment Runner.

Defines ExperimentSpec, ExperimentStatus, and ExperimentRecord — the
data contracts that bind together configuration, execution state, and all
collected artifacts for a single named experiment.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import ConfigDict, Field, model_validator

from llm_reliability.utils.serialization import SerializableModel


class ExperimentState(str, Enum):
    """Lifecycle states for a managed experiment."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class BenchmarkSpec(SerializableModel):
    """Specification for one benchmark within an experiment."""

    name: str = Field(min_length=1, description="Registered benchmark name.")
    dataset_path: str = Field(min_length=1, description="Path to dataset JSON file.")
    adapter_metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSpec(SerializableModel):
    """Specification for one agent within an experiment."""

    name: str = Field(min_length=1, description="Registered agent class name.")
    agent_metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentSpec(SerializableModel):
    """Full specification for a multi-benchmark, multi-agent experiment run.

    ExperimentSpec is the top-level input to ExperimentRunner. It captures
    every parameter needed to reproduce the experiment deterministically.
    """

    experiment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this experiment.",
    )
    experiment_name: str = Field(min_length=1)
    benchmarks: list[BenchmarkSpec] = Field(min_length=1)
    agents: list[AgentSpec] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1, description="Seeds for repeated runs.")
    repetitions: int = Field(default=1, gt=0)
    perturbations: list[str] = Field(default_factory=list)
    fault_injection: bool = False
    parallel: bool = Field(default=False, description="Enable parallel task execution.")
    max_workers: int = Field(default=4, gt=0)
    output_dir: str = Field(default="results", description="Root output directory.")
    llm: str = Field(default="mock", min_length=1)
    prompt_version: str = Field(default="1", min_length=1)
    dataset_version: str = Field(default="1", min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _no_duplicate_benchmarks(self) -> "ExperimentSpec":
        names = [b.name for b in self.benchmarks]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate benchmark names are not allowed.")
        return self

    @model_validator(mode="after")
    def _no_duplicate_agents(self) -> "ExperimentSpec":
        names = [a.name for a in self.agents]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate agent names are not allowed.")
        return self

    @model_validator(mode="after")
    def _validate_seeds(self) -> "ExperimentSpec":
        for s in self.seeds:
            if s < 0:
                raise ValueError(f"Seeds must be non-negative integers, got {s}.")
        return self


class ExperimentStatus(SerializableModel):
    """Runtime tracking state for a running experiment."""

    model_config = ConfigDict(
        frozen=False,
        extra="forbid",
    )

    experiment_id: str
    state: ExperimentState = ExperimentState.PENDING
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    current_benchmark: str | None = None
    current_agent: str | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)

    def progress_fraction(self) -> float:
        """Return completion fraction in [0, 1]."""
        if self.total_runs == 0:
            return 0.0
        return self.completed_runs / self.total_runs
