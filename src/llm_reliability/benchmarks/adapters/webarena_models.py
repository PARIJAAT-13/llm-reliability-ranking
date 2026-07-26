from __future__ import annotations

from typing import Any

from pydantic import Field

from llm_reliability.utils.serialization import SerializableModel


class WebArenaTask(SerializableModel):
    task_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    task_category: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebArenaResult(SerializableModel):
    task_id: str = Field(min_length=1)
    agent_output: str
    success: bool
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebArenaMetadata(SerializableModel):
    name: str = "WebArena"
    version: str = "1.0"
    task_count: int = Field(ge=0)
    deterministic: bool = True
