"""
Strongly-typed Pydantic models for LLM request payloads.

LLMRequest captures all provider-agnostic parameters for a single inference call.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from llm_reliability.agents.adapters.exceptions import RequestValidationError
from llm_reliability.utils.serialization import SerializableModel


class LLMRequest(SerializableModel):
    """Provider-agnostic model for a single LLM inference request."""

    prompt: str = Field(min_length=1, description="The user prompt to send to the model.")
    system_prompt: str | None = Field(
        default=None, description="Optional system-level instructions."
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature.")
    max_tokens: int = Field(default=1024, gt=0, description="Maximum tokens to generate.")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling probability.")
    seed: int | None = Field(default=None, description="Optional seed for deterministic sampling.")
    stop_sequences: list[str] = Field(default_factory=list, description="Optional stop tokens.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Caller-supplied metadata.")

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise RequestValidationError("prompt must not be blank.")
        return v
