"""
Purpose
-------
Provide strongly-typed data models for the GAIA benchmark.

Responsibilities
----------------
- Define the GAIATask schema (question, ground_truth_answer, difficulty, etc.)
- Define the GAIAResult schema
- Define the GAIAMetadata schema
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from llm_reliability.utils.serialization import SerializableModel


class GAIATask(SerializableModel):
    """Schema for a single GAIA task."""

    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    ground_truth_answer: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    task_category: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GAIAResult(SerializableModel):
    """Schema for a GAIA execution result."""

    task_id: str = Field(min_length=1)
    agent_output: str
    success: bool
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GAIAMetadata(SerializableModel):
    """Schema for GAIA benchmark metadata."""

    name: str = "GAIA"
    version: str = "1.0"
    task_count: int = Field(ge=0)
    deterministic: bool = True
