"""
Purpose
-------
Capture the complete, immutable audit trail for a single agent task execution.

Responsibilities
----------------
- Store configuration hash, seed, and identity fields for reproducibility
- Record runtime telemetry, stdout/stderr, status, and errors
- Preserve software versions and environment metadata
- Support canonical serialization without mutation

Usage example
-------------
>>> from llm_reliability.records import ExecutionRecord
>>> record = ExecutionRecord(
...     configuration_hash="abc123",
...     seed=42,
...     benchmark="mock",
...     agent="mock_agent",
...     task_id="task-1",
...     run_index=0,
...     runtime_seconds=1.5,
...     timestamp="2026-01-01T00:00:00+00:00",
...     stdout="ok",
...     stderr="",
...     status="success",
... )
>>> record.sha256()

Design notes
------------
ExecutionRecord is the leaf artifact produced by Benchmark.run. It is never
modified after creation. EvaluationRecord derives exclusively from this type.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from llm_reliability.utils.serialization import SerializableModel

ExecutionStatus = Literal["success", "failure", "error", "timeout"]


class ExecutionRecord(SerializableModel):
    """Immutable record of a single agent execution on one benchmark task."""

    configuration_hash: str = Field(min_length=64, max_length=64)
    seed: int = Field(ge=0)
    benchmark: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    run_index: int = Field(ge=0)
    perturbation: str | None = None
    fault_injected: bool = False
    runtime_seconds: float = Field(ge=0.0)
    timestamp: str = Field(min_length=1)
    stdout: str = ""
    stderr: str = ""
    status: ExecutionStatus
    error: str | None = None
    agent_output: Any = None
    software_versions: dict[str, str] = Field(default_factory=dict)
    environment_metadata: dict[str, Any] = Field(default_factory=dict)
