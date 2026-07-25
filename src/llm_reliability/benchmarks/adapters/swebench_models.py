"""
Purpose
-------
Provide strongly-typed data models for the SWE-bench Lite benchmark.

Responsibilities
----------------
- Define the SWEBenchTask schema
- Define the SWEBenchResult schema
- Define the SWEBenchMetadata schema
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from llm_reliability.utils.serialization import SerializableModel


class SWEBenchTask(SerializableModel):
    """Schema for a single SWE-bench Lite task."""

    task_id: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    patch: str = Field(min_length=1)
    expected_outcome: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SWEBenchResult(SerializableModel):
    """Schema for a SWE-bench Lite execution result."""

    task_id: str = Field(min_length=1)
    agent_output: str
    success: bool
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SWEBenchMetadata(SerializableModel):
    """Schema for SWE-bench Lite benchmark metadata."""

    name: str = "SWE-bench Lite"
    version: str = "1.0"
    task_count: int = Field(ge=0)
    deterministic: bool = True
