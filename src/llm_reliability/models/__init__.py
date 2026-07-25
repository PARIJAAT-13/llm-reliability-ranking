from __future__ import annotations

from llm_reliability.models.discovery import discover_local_models, merge_discovered
from llm_reliability.models.model_info import ModelInfo
from llm_reliability.models.model_registry import (
    DuplicateModelError,
    ModelNotFoundError,
    ModelRegistry,
    ValidationError,
)
from llm_reliability.models.ollama_models import SUPPORTED_OLLAMA_MODELS

__all__ = [
    "ModelInfo",
    "ModelRegistry",
    "DuplicateModelError",
    "ModelNotFoundError",
    "ValidationError",
    "SUPPORTED_OLLAMA_MODELS",
    "discover_local_models",
    "merge_discovered",
]


def populate_registry() -> int:
    ModelRegistry.reset()
    ModelRegistry.register_many(SUPPORTED_OLLAMA_MODELS)
    return ModelRegistry.count()
