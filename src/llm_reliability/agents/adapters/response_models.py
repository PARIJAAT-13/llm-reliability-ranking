"""
Strongly-typed Pydantic models for LLM response payloads.

LLMResponse standardizes output across all providers so downstream
components never need to handle provider-specific response structures.
"""

from typing import Any

from pydantic import Field

from llm_reliability.utils.serialization import SerializableModel


class LLMResponse(SerializableModel):
    """Provider-agnostic model for a single LLM inference response."""

    text: str = Field(description="The generated text from the model.")
    finish_reason: str = Field(description="Why generation stopped (stop, length, error, etc.).")
    latency_ms: float = Field(ge=0.0, description="Wall-clock latency in milliseconds.")
    tokens_input: int = Field(ge=0, description="Number of tokens in the input prompt.")
    tokens_output: int = Field(ge=0, description="Number of tokens in the generated output.")
    model_name: str = Field(min_length=1, description="Identifier of the model used.")
    provider: str = Field(min_length=1, description="Name of the LLM provider.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Provider-specific metadata.")
