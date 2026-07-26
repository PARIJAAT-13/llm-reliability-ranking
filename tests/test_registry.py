"""Tests for BenchmarkRegistry."""

import pytest

from llm_reliability.benchmarks.adapters.base_adapter import \
    BaseBenchmarkAdapter
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry


class ValidAdapter1(BaseBenchmarkAdapter):
    def _load_tasks(self):
        pass

    def run(self, agent, task):
        pass

    def evaluate(self, execution):
        pass


class ValidAdapter2(BaseBenchmarkAdapter):
    def _load_tasks(self):
        pass

    def run(self, agent, task):
        pass

    def evaluate(self, execution):
        pass


class InvalidAdapter:
    pass


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is empty before and after each test."""
    BenchmarkRegistry._adapters.clear()
    yield
    BenchmarkRegistry._adapters.clear()


def test_registry_registration():
    BenchmarkRegistry.register("adapter1", ValidAdapter1)
    assert BenchmarkRegistry.exists("adapter1")
    assert BenchmarkRegistry.get("adapter1") is ValidAdapter1


def test_registry_duplicate_registration():
    BenchmarkRegistry.register("adapter1", ValidAdapter1)
    with pytest.raises(ValueError, match="is already registered"):
        BenchmarkRegistry.register("adapter1", ValidAdapter2)


def test_registry_invalid_adapter():
    with pytest.raises(TypeError, match="must be a subclass of BaseBenchmarkAdapter"):
        BenchmarkRegistry.register("invalid", InvalidAdapter)


def test_registry_lookup():
    BenchmarkRegistry.register("a2", ValidAdapter2)
    BenchmarkRegistry.register("a1", ValidAdapter1)

    adapters = BenchmarkRegistry.list()
    assert adapters == ["a1", "a2"]  # Ordered

    with pytest.raises(ValueError, match="not found in registry"):
        BenchmarkRegistry.get("unknown")


def test_registry_removal():
    BenchmarkRegistry.register("adapter1", ValidAdapter1)
    assert BenchmarkRegistry.exists("adapter1")

    BenchmarkRegistry.unregister("adapter1")
    assert not BenchmarkRegistry.exists("adapter1")

    with pytest.raises(ValueError, match="is not registered"):
        BenchmarkRegistry.unregister("adapter1")
