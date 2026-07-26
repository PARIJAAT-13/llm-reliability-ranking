"""Extended tests for AgentFactory — prefix resolution, all agent types, and edge cases."""

from __future__ import annotations

import pytest

from llm_reliability.agents.agent_factory import AgentFactory
from llm_reliability.runtime.interface import Runtime


class _MockConfig:
    def __init__(self):
        self.model = "mock-model"
        self.temperature = 0.0
        self.max_tokens = 100
        self.api_key = "mock-key"


@pytest.fixture(autouse=True)
def discover_runtimes():
    from llm_reliability.runtime.registry import RuntimeRegistry

    RuntimeRegistry.discover()
    yield


class TestAgentFactoryResolve:
    def test_resolve_openrouter(self):
        result = AgentFactory.resolve("openrouter")
        assert result is not None

    def test_resolve_together(self):
        result = AgentFactory.resolve("together")
        assert result is not None

    def test_resolve_groq(self):
        result = AgentFactory.resolve("groq")
        assert result is not None

    def test_resolve_fireworks(self):
        result = AgentFactory.resolve("fireworks")
        assert result is not None

    def test_resolve_cohere(self):
        result = AgentFactory.resolve("cohere")
        assert result is not None

    def test_resolve_mistral(self):
        result = AgentFactory.resolve("mistral")
        assert result is not None

    def test_resolve_xai(self):
        result = AgentFactory.resolve("xai")
        assert result is not None

    def test_resolve_grok(self):
        result = AgentFactory.resolve("grok")
        assert result is not None

    def test_resolve_perplexity(self):
        result = AgentFactory.resolve("perplexity")
        assert result is not None

    def test_resolve_sonar(self):
        result = AgentFactory.resolve("sonar")
        assert result is not None

    def test_resolve_azure(self):
        result = AgentFactory.resolve("azure")
        assert result is not None

    def test_resolve_azure_openai(self):
        result = AgentFactory.resolve("azure_openai")
        assert result is not None

    def test_resolve_bedrock(self):
        result = AgentFactory.resolve("bedrock")
        assert result is not None

    def test_resolve_aws(self):
        result = AgentFactory.resolve("aws")
        assert result is not None

    def test_resolve_vertex(self):
        result = AgentFactory.resolve("vertex")
        assert result is not None

    def test_resolve_vertexai(self):
        result = AgentFactory.resolve("vertexai")
        assert result is not None

    def test_resolve_sambanova(self):
        result = AgentFactory.resolve("sambanova")
        assert result is not None

    def test_resolve_cerebras(self):
        result = AgentFactory.resolve("cerebras")
        assert result is not None

    def test_resolve_nim(self):
        result = AgentFactory.resolve("nim")
        assert result is not None

    def test_resolve_nvidia(self):
        result = AgentFactory.resolve("nvidia")
        assert result is not None

    def test_resolve_litellm(self):
        result = AgentFactory.resolve("litellm")
        assert result is not None

    def test_resolve_sglang(self):
        result = AgentFactory.resolve("sglang")
        assert result is not None

    def test_resolve_vllm(self):
        result = AgentFactory.resolve("vllm")
        assert result is not None

    def test_resolve_huggingface(self):
        result = AgentFactory.resolve("huggingface")
        assert result is not None

    def test_resolve_hf(self):
        result = AgentFactory.resolve("hf")
        assert result is not None

    def test_resolve_llamacpp(self):
        result = AgentFactory.resolve("llamacpp")
        assert result is not None

    def test_resolve_llama_cpp(self):
        result = AgentFactory.resolve("llama.cpp")
        assert result is not None

    def test_resolve_unknown_returns_none(self):
        result = AgentFactory.resolve("completely_unknown_agent_xyz")
        assert result is None

    def test_resolve_empty_string_returns_none(self):
        result = AgentFactory.resolve("")
        assert result is None


class TestAgentFactoryCreate:
    def test_create_mock_agent(self):
        agent = AgentFactory.create("mock", _MockConfig())
        assert isinstance(agent, Runtime)
        result = agent.run({"expected_answer": "mock response"})
        assert result == "mock response"

    def test_create_mock_agent_via_alias(self):
        agent = AgentFactory.create("mock_agent", _MockConfig())
        assert isinstance(agent, Runtime)

    def test_create_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown agent"):
            AgentFactory.create("no_such_agent_ever", _MockConfig())

    def test_create_with_colon_prefix(self):
        agent = AgentFactory.create("mock:my-model", _MockConfig())
        assert isinstance(agent, Runtime)

    def test_create_case_insensitive(self):
        agent = AgentFactory.create("Mock", _MockConfig())
        assert isinstance(agent, Runtime)


class TestAgentFactoryIsMock:
    def test_is_mock_returns_true_for_mock(self):
        assert AgentFactory.is_mock("mock") is True

    def test_is_mock_returns_true_for_mock_agent(self):
        assert AgentFactory.is_mock("mock_agent") is True

    def test_is_mock_returns_false_for_gpt(self):
        assert AgentFactory.is_mock("gpt") is False

    def test_is_mock_returns_false_for_openrouter(self):
        assert AgentFactory.is_mock("openrouter") is False

    def test_is_mock_returns_false_for_together(self):
        assert AgentFactory.is_mock("together") is False

    def test_is_mock_returns_false_for_groq(self):
        assert AgentFactory.is_mock("groq") is False

    def test_is_mock_returns_false_for_fireworks(self):
        assert AgentFactory.is_mock("fireworks") is False

    def test_is_mock_returns_false_for_cohere(self):
        assert AgentFactory.is_mock("cohere") is False

    def test_is_mock_returns_false_for_mistral(self):
        assert AgentFactory.is_mock("mistral") is False

    def test_is_mock_returns_false_for_xai(self):
        assert AgentFactory.is_mock("xai") is False

    def test_is_mock_returns_false_for_perplexity(self):
        assert AgentFactory.is_mock("perplexity") is False

    def test_is_mock_returns_false_for_azure(self):
        assert AgentFactory.is_mock("azure") is False

    def test_is_mock_returns_false_for_bedrock(self):
        assert AgentFactory.is_mock("bedrock") is False

    def test_is_mock_returns_false_for_vertex(self):
        assert AgentFactory.is_mock("vertex") is False

    def test_is_mock_returns_false_for_sambanova(self):
        assert AgentFactory.is_mock("sambanova") is False

    def test_is_mock_returns_false_for_cerebras(self):
        assert AgentFactory.is_mock("cerebras") is False

    def test_is_mock_returns_false_for_nim(self):
        assert AgentFactory.is_mock("nim") is False

    def test_is_mock_returns_false_for_litellm(self):
        assert AgentFactory.is_mock("litellm") is False

    def test_is_mock_returns_false_for_sglang(self):
        assert AgentFactory.is_mock("sglang") is False

    def test_is_mock_for_unknown_name(self):
        assert AgentFactory.is_mock("unknown_agent") is False


class TestAgentFactoryAvailableNames:
    def test_available_names_contains_mock(self):
        names = AgentFactory.available_names()
        assert "mock" in names

    def test_available_names_contains_openrouter(self):
        names = AgentFactory.available_names()
        assert "openrouter" in names

    def test_available_names_contains_together(self):
        names = AgentFactory.available_names()
        assert "together" in names

    def test_available_names_contains_groq(self):
        names = AgentFactory.available_names()
        assert "groq" in names

    def test_available_names_contains_fireworks(self):
        names = AgentFactory.available_names()
        assert "fireworks" in names

    def test_available_names_contains_cohere(self):
        names = AgentFactory.available_names()
        assert "cohere" in names

    def test_available_names_contains_mistral(self):
        names = AgentFactory.available_names()
        assert "mistral" in names

    def test_available_names_contains_xai(self):
        names = AgentFactory.available_names()
        assert "xai" in names

    def test_available_names_contains_perplexity(self):
        names = AgentFactory.available_names()
        assert "perplexity" in names

    def test_available_names_contains_bedrock(self):
        names = AgentFactory.available_names()
        assert "bedrock" in names

    def test_available_names_contains_vertex(self):
        names = AgentFactory.available_names()
        assert "vertex" in names

    def test_available_names_contains_sambanova(self):
        names = AgentFactory.available_names()
        assert "sambanova" in names

    def test_available_names_contains_cerebras(self):
        names = AgentFactory.available_names()
        assert "cerebras" in names

    def test_available_names_contains_nim(self):
        names = AgentFactory.available_names()
        assert "nim" in names

    def test_available_names_contains_litellm(self):
        names = AgentFactory.available_names()
        assert "litellm" in names

    def test_available_names_contains_sglang(self):
        names = AgentFactory.available_names()
        assert "sglang" in names

    def test_available_names_sorted(self):
        names = AgentFactory.available_names()
        assert names == sorted(names)
