"""
Provider Registry for the LLM Agent Adapter Framework.

The registry manages a global mapping from provider names to adapter classes,
enabling the pipeline to instantiate providers by name without hard-coded imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_reliability.agents.adapters.base_llm_adapter import BaseLLMAdapter


class ProviderRegistry:
    """Registry for LLM provider adapters."""

    _adapters: dict[str, type[BaseLLMAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_cls: type[BaseLLMAdapter]) -> None:
        """Register a provider adapter by name.

        Raises ValueError if the name is already registered.
        Raises TypeError if adapter_cls is not a subclass of BaseLLMAdapter.
        """
        # Import here to avoid circular imports at module load time
        from llm_reliability.agents.adapters.base_llm_adapter import \
            BaseLLMAdapter

        if name in cls._adapters:
            raise ValueError(f"Provider '{name}' is already registered.")
        if not issubclass(adapter_cls, BaseLLMAdapter):
            raise TypeError("adapter_cls must be a subclass of BaseLLMAdapter.")
        cls._adapters[name] = adapter_cls

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a provider adapter by name.

        Raises ValueError if the name is not registered.
        """
        if name not in cls._adapters:
            raise ValueError(f"Provider '{name}' is not registered.")
        del cls._adapters[name]

    @classmethod
    def get(cls, name: str) -> type[BaseLLMAdapter]:
        """Return the adapter class for the given provider name.

        Raises ValueError if the provider is not registered.
        """
        if name not in cls._adapters:
            raise ValueError(f"Provider '{name}' not found in registry.")
        return cls._adapters[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return a sorted list of all registered provider names."""
        return sorted(cls._adapters.keys())

    @classmethod
    def exists(cls, name: str) -> bool:
        """Return True if the provider is registered."""
        return name in cls._adapters
