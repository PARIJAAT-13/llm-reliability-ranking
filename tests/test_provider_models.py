from __future__ import annotations

import pytest

from llm_reliability.models import populate_registry
from llm_reliability.models.model_registry import ModelRegistry
from llm_reliability.models.provider_models import SUPPORTED_PROVIDER_MODELS


@pytest.fixture(autouse=True)
def reset_registry():
    ModelRegistry.clear()
    yield
    ModelRegistry.clear()


class TestSupportedProviderModels:
    def test_count_at_least_48(self):
        assert len(SUPPORTED_PROVIDER_MODELS) >= 48

    def test_all_have_required_fields(self):
        for m in SUPPORTED_PROVIDER_MODELS:
            assert m.provider
            assert m.runtime
            assert m.name

    def test_all_have_ollama_identifier_with_colon(self):
        for m in SUPPORTED_PROVIDER_MODELS:
            assert ":" in m.ollama_identifier
            assert m.ollama_identifier.startswith(f"{m.provider}:")

    def test_unique_ollama_identifiers(self):
        ids = [m.ollama_identifier for m in SUPPORTED_PROVIDER_MODELS]
        assert len(ids) == len(set(ids))

    def test_all_providers_have_non_empty_family(self):
        for m in SUPPORTED_PROVIDER_MODELS:
            assert m.family

    def test_non_zero_context_window(self):
        for m in SUPPORTED_PROVIDER_MODELS:
            assert m.context_window > 0

    def test_multiple_providers_present(self):
        providers = {m.provider for m in SUPPORTED_PROVIDER_MODELS}
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
        assert "deepseek" in providers
        assert "mistral" in providers
        assert "azure_openai" in providers
        assert "bedrock" in providers
        assert "vertex" in providers

    def test_openai_models_specific_count(self):
        openai_models = [m for m in SUPPORTED_PROVIDER_MODELS if m.provider == "openai"]
        assert len(openai_models) >= 7


class TestPopulateRegistry:
    def test_populate_registry_returns_at_least_100(self):
        count = populate_registry()
        assert count >= 100

    def test_populate_count_matches_ollama_plus_provider(self):
        from llm_reliability.models.ollama_models import \
            SUPPORTED_OLLAMA_MODELS

        expected = len(SUPPORTED_OLLAMA_MODELS) + len(SUPPORTED_PROVIDER_MODELS)
        count = populate_registry()
        assert count == expected

    def test_count_method_consistent(self):
        populate_registry()
        assert ModelRegistry.count() >= 100

    def test_provider_models_accessible_by_identifier(self):
        populate_registry()
        for m in SUPPORTED_PROVIDER_MODELS:
            retrieved = ModelRegistry.get(m.ollama_identifier)
            assert retrieved.name == m.name

    def test_list_identifiers_includes_provider_prefixes(self):
        populate_registry()
        ids = ModelRegistry.list_identifiers()
        provider_prefixes = {m.ollama_identifier for m in SUPPORTED_PROVIDER_MODELS}
        for pref in provider_prefixes:
            assert pref in ids

    def test_exists_returns_true_for_provider_models(self):
        populate_registry()
        assert ModelRegistry.exists("openai:gpt-4o")
        assert ModelRegistry.exists("anthropic:claude-3-5-sonnet-20241022")

    def test_exists_returns_false_for_unknown(self):
        populate_registry()
        assert not ModelRegistry.exists("nonexistent:foo")

    def test_list_by_family_includes_new_families(self):
        populate_registry()
        families = ModelRegistry.list_by_family()
        assert "OpenAI" in families
        assert "Anthropic" in families
        assert "Google" in families
        assert "Llama" in families
        assert "Mistral" in families

    def test_list_families_contains_all(self):
        populate_registry()
        families = ModelRegistry.list_families()
        assert "OpenAI" in families
        assert "Anthropic" in families
        assert "AWS Bedrock" in families

    def test_to_dict_list_works(self):
        populate_registry()
        dicts = ModelRegistry.to_dict_list()
        assert len(dicts) >= 100
        first = dicts[0]
        assert "ollama_identifier" in first
        assert "name" in first
        assert "provider" in first
        assert "runtime" in first
        assert "family" in first

    def test_to_dict_list_all_entries_have_required_keys(self):
        populate_registry()
        dicts = ModelRegistry.to_dict_list()
        required = {"ollama_identifier", "name", "provider", "runtime", "family", "parameters"}
        for d in dicts:
            assert required.issubset(d.keys()), f"Missing keys in {d.get('ollama_identifier', '?')}"
