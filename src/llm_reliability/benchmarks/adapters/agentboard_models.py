"""
Purpose
-------
Provide strongly-typed data models for the AgentBoard benchmark.

Responsibilities
----------------
- Define the AgentBoardTask schema
- Define the AgentBoardResult schema
- Define the AgentBoardMetadata schema
"""

from typing import Any

from pydantic import Field

from llm_reliability.utils.serialization import SerializableModel


class AgentBoardTask(SerializableModel):
    """Schema for a single AgentBoard task."""

    task_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    expected_output: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    category: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentBoardResult(SerializableModel):
    """Schema for an AgentBoard execution result."""

    task_id: str = Field(min_length=1)
    agent_output: str
    success: bool
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentBoardMetadata(SerializableModel):
    """Schema for AgentBoard benchmark metadata."""

    name: str = "AgentBoard"
    version: str = "1.0"
    task_count: int = Field(ge=0)
    deterministic: bool = True
