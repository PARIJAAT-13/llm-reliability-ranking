"""Tests for AgentFactory — agent resolution and creation."""

import pytest

from llm_reliability.agents.agent_factory import AgentFactory
from llm_reliability.runtime.interface import Runtime
from llm_reliability.runtime.registry import RuntimeRegistry


@pytest.fixture(autouse=True)
def discover_runtimes():
    RuntimeRegistry.discover()
    yield


class _MockConfig:
    def __init__(self):
        self.model = "mock-model"
        self.temperature = 0.0
        self.max_tokens = 100
        self.api_key = "mock-key"


class TestAgentFactory:
    def test_create_mock_agent(self):
        agent = AgentFactory.create("mock", _MockConfig())
        assert isinstance(agent, Runtime)
        result = agent.run({"expected_answer": "mock response"})
        assert isinstance(result, str)

    def test_available_names_returns_sorted(self):
        names = AgentFactory.available_names()
        assert isinstance(names, list)
        assert names == sorted(names)
        assert "mock" in names

    def test_create_unknown_agent_raises(self):
        with pytest.raises(ValueError, match="Unknown agent"):
            AgentFactory.create("completely_unknown_agent", _MockConfig())

    def test_resolve_known_runtime(self):
        result = AgentFactory.resolve("mock")
        assert result is not None

    def test_resolve_unknown_returns_none(self):
        result = AgentFactory.resolve("no_such_provider")
        assert result is None

    def test_is_mock_returns_true_for_mock(self):
        assert AgentFactory.is_mock("mock") is True

    def test_is_mock_returns_false_for_non_mock(self):
        assert AgentFactory.is_mock("gpt") is False
