from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ModelInfo(BaseModel):
    family: str = Field(description="Model family name, e.g. Llama, Qwen, Mistral")
    name: str = Field(description="Human-readable display name, e.g. Llama 3.1 8B")
    parameters: str = Field(description="Parameter count string, e.g. 8B, 70B, 405B")
    parameter_count: float = Field(ge=0, description="Numeric parameter count in billions")
    context_window: int = Field(ge=0, default=0, description="Maximum context length in tokens")
    architecture: str = Field(
        default="Transformer (Decoder-Only)", description="Model architecture"
    )
    quantization: str | None = Field(default=None, description="Default quantization level")
    recommended_ram_gb: float | None = Field(
        default=None, ge=0, description="Recommended system RAM in GB"
    )
    recommended_vram_gb: float | None = Field(
        default=None, ge=0, description="Recommended GPU VRAM in GB"
    )
    ollama_identifier: str = Field(
        min_length=1, description="Ollama pull identifier, e.g. llama3.1:8b"
    )
    provider: Literal["ollama"] = Field(default="ollama", description="Inference provider")
    runtime: Literal["ollama"] = Field(default="ollama", description="Runtime name")
    status: Literal["supported", "experimental", "deprecated"] = Field(
        default="supported", description="Support status"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

    @model_validator(mode="after")
    def validate_identifier_format(self) -> ModelInfo:
        if ":" not in self.ollama_identifier:
            raise ValueError(
                f"ollama_identifier must contain a colon (e.g. 'llama3.1:8b'), got '{self.ollama_identifier}'"
            )
        return self
