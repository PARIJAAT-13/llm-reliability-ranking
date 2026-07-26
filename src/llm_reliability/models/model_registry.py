"""Central registry for supported LLM model configurations."""

from __future__ import annotations

from typing import Any

from llm_reliability.models.model_info import ModelInfo


class DuplicateModelError(ValueError):
    pass


class ModelNotFoundError(KeyError):
    pass


class ValidationError(ValueError):
    pass


class ModelRegistry:
    _models: dict[str, ModelInfo] = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls) -> None:
        if cls._initialized:
            return
        cls._models = {}
        cls._initialized = True

    @classmethod
    def reset(cls) -> None:
        cls._models = {}
        cls._initialized = True

    @classmethod
    def clear(cls) -> None:
        cls._models = {}
        cls._initialized = False

    @classmethod
    def register(cls, model: ModelInfo) -> None:
        cls._ensure_initialized()
        if not isinstance(model, ModelInfo):
            raise TypeError(f"Expected ModelInfo instance, got {type(model).__name__}")
        identifier = model.ollama_identifier
        if identifier in cls._models:
            existing = cls._models[identifier]
            raise DuplicateModelError(
                f"Duplicate model identifier '{identifier}': '{existing.name}' already registered."
            )
        cls._models[identifier] = model

    @classmethod
    def register_many(cls, models: list[ModelInfo]) -> None:
        errors: list[str] = []
        for model in models:
            try:
                cls.register(model)
            except DuplicateModelError as e:
                errors.append(str(e))
        if errors:
            raise DuplicateModelError("\n".join(errors))

    @classmethod
    def get(cls, identifier: str) -> ModelInfo:
        cls._ensure_initialized()
        if identifier not in cls._models:
            raise ModelNotFoundError(
                f"Model '{identifier}' not found in registry. "
                f"Available: {sorted(cls._models.keys())}"
            )
        return cls._models[identifier]

    @classmethod
    def exists(cls, identifier: str) -> bool:
        cls._ensure_initialized()
        return identifier in cls._models

    @classmethod
    def list_identifiers(cls) -> list[str]:
        cls._ensure_initialized()
        return sorted(cls._models.keys())

    @classmethod
    def list_models(cls) -> list[ModelInfo]:
        cls._ensure_initialized()
        return sorted(cls._models.values(), key=lambda m: m.ollama_identifier)

    @classmethod
    def list_by_family(cls) -> dict[str, list[ModelInfo]]:
        cls._ensure_initialized()
        result: dict[str, list[ModelInfo]] = {}
        for model in cls._models.values():
            result.setdefault(model.family, []).append(model)
        for family in result:
            result[family].sort(key=lambda m: m.parameter_count)
        return result

    @classmethod
    def list_families(cls) -> list[str]:
        cls._ensure_initialized()
        return sorted({m.family for m in cls._models.values()})

    @classmethod
    def count(cls) -> int:
        cls._ensure_initialized()
        return len(cls._models)

    @classmethod
    def unregister(cls, identifier: str) -> None:
        cls._ensure_initialized()
        if identifier not in cls._models:
            raise ModelNotFoundError(f"Model '{identifier}' not found.")
        del cls._models[identifier]

    @classmethod
    def to_dict_list(cls) -> list[dict[str, Any]]:
        cls._ensure_initialized()
        return [m.model_dump() for m in cls._models.values()]

    @classmethod
    def _ensure_initialized(cls) -> None:
        if not cls._initialized:
            cls.initialize()
