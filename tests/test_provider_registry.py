"""Tests for ProviderRegistry."""

import pytest

from llm_reliability.agents.adapters.base_llm_adapter import BaseLLMAdapter
from llm_reliability.agents.adapters.provider_registry import ProviderRegistry
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse


class Provider1(BaseLLMAdapter):
    def initialize(self): pass
    def generate(self, r: LLMRequest) -> LLMResponse: ...
    def shutdown(self): pass
    def provider_metadata(self): return {}
    def health_check(self): return True


class Provider2(BaseLLMAdapter):
    def initialize(self): pass
    def generate(self, r: LLMRequest) -> LLMResponse: ...
    def shutdown(self): pass
    def provider_metadata(self): return {}
    def health_check(self): return True


class NotAProvider:
    pass


@pytest.fixture(autouse=True)
def clean_registry():
    ProviderRegistry._adapters.clear()
    yield
    ProviderRegistry._adapters.clear()


def test_registry_registration():
    ProviderRegistry.register("p1", Provider1)
    assert ProviderRegistry.exists("p1")
    assert ProviderRegistry.get("p1") is Provider1


def test_registry_duplicate_raises():
    ProviderRegistry.register("p1", Provider1)
    with pytest.raises(ValueError, match="already registered"):
        ProviderRegistry.register("p1", Provider2)


def test_registry_invalid_type_raises():
    with pytest.raises(TypeError, match="subclass of BaseLLMAdapter"):
        ProviderRegistry.register("bad", NotAProvider)  # type: ignore


def test_registry_lookup_missing_raises():
    with pytest.raises(ValueError, match="not found"):
        ProviderRegistry.get("unknown")


def test_registry_list_sorted():
    ProviderRegistry.register("z_provider", Provider2)
    ProviderRegistry.register("a_provider", Provider1)
    assert ProviderRegistry.list() == ["a_provider", "z_provider"]


def test_registry_unregister():
    ProviderRegistry.register("p1", Provider1)
    ProviderRegistry.unregister("p1")
    assert not ProviderRegistry.exists("p1")


def test_registry_unregister_missing_raises():
    with pytest.raises(ValueError, match="not registered"):
        ProviderRegistry.unregister("nonexistent")
