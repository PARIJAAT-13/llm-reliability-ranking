"""
Purpose
-------
Provide a global registry for all benchmark adapters.

Responsibilities
----------------
- Register adapters by name
- Unregister adapters
- Retrieve adapters by name safely
- List all available adapters
- Prevent duplicate registrations
"""

from typing import Type

from llm_reliability.benchmarks.adapters.base_adapter import BaseBenchmarkAdapter


class BenchmarkRegistry:
    """Registry to manage and discover benchmark adapters dynamically."""

    _adapters: dict[str, type[BaseBenchmarkAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_cls: type[BaseBenchmarkAdapter]) -> None:
        """Register a benchmark adapter by name."""
        if name in cls._adapters:
            raise ValueError(f"Adapter '{name}' is already registered.")
        if not issubclass(adapter_cls, BaseBenchmarkAdapter):
            raise TypeError("adapter_cls must be a subclass of BaseBenchmarkAdapter.")
        cls._adapters[name] = adapter_cls

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a benchmark adapter by name."""
        if name not in cls._adapters:
            raise ValueError(f"Adapter '{name}' is not registered.")
        del cls._adapters[name]

    @classmethod
    def get(cls, name: str) -> type[BaseBenchmarkAdapter]:
        """Retrieve a registered benchmark adapter by name."""
        if name not in cls._adapters:
            raise ValueError(f"Adapter '{name}' not found in registry.")
        return cls._adapters[name]

    @classmethod
    def list(cls) -> list[str]:
        """List all available benchmark adapter names."""
        return sorted(list(cls._adapters.keys()))

    @classmethod
    def exists(cls, name: str) -> bool:
        """Check if an adapter is registered."""
        return name in cls._adapters
