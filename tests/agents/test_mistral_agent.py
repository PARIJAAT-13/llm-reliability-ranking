from __future__ import annotations

import sys
from collections.abc import Generator
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_reliability.agents.adapters.exceptions import AuthenticationError
from llm_reliability.configs.config import Configuration


def _make_config(**overrides: Any) -> Configuration:
    defaults: dict[str, Any] = dict(
        experiment_name="mistral_test",
        benchmark="AgentBoard",
        agent="MistralAgent",
        llm="mistral-large-2407",
        prompt_version="v1",
        dataset_version="1.0",
        seed=42,
        repetitions=1,
    )
    defaults.update(overrides)
    return Configuration(**defaults)


def _make_mistral_completion(text: str = "answer", finish_reason: str = "stop") -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5

    message = MagicMock()
    message.content = text

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _stub_mistralai_module() -> ModuleType:
    mistralai_mock = ModuleType("mistralai")

    client_mock = MagicMock()
    mistralai_mock.Mistral = MagicMock(return_value=client_mock)
    return mistralai_mock


@pytest.fixture
def mistralai_mod() -> Generator[ModuleType, None, None]:
    stub = _stub_mistralai_module()
    with patch.dict(sys.modules, {"mistralai": stub}):
        yield stub


@pytest.fixture
def config() -> Configuration:
    return _make_config()


@pytest.fixture
def clear_registries():
    from llm_reliability.agents.adapters.provider_registry import \
        ProviderRegistry
    from llm_reliability.runtime.registry import RuntimeRegistry

    ProviderRegistry._adapters.clear()
    RuntimeRegistry._runtimes.clear()
    RuntimeRegistry._initialised = False
    RuntimeRegistry._discovered_module_names.clear()


@pytest.fixture
def agent_with_mock_mistral(config, mistralai_mod, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    from llm_reliability.agents.mistral_agent import MistralAgent

    agent = MistralAgent(config)
    agent.initialize()
    return agent


class TestMistralAgent:
    @pytest.mark.usefixtures("clear_registries")
    def test_initialize_raises_without_api_key(self, config, mistralai_mod, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        from llm_reliability.agents.mistral_agent import MistralAgent

        agent = MistralAgent(config)
        with pytest.raises(AuthenticationError, match="MISTRAL_API_KEY"):
            agent.initialize()

    @pytest.mark.usefixtures("clear_registries")
    def test_initialize_succeeds_with_api_key(self, config, mistralai_mod, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        from llm_reliability.agents.mistral_agent import MistralAgent

        agent = MistralAgent(config)
        agent.initialize()
        assert agent._adapter._client is not None

    @pytest.mark.usefixtures("clear_registries")
    def test_run_returns_text(self, agent_with_mock_mistral, mistralai_mod):
        response = _make_mistral_completion(text="four")
        mistralai_mod.Mistral.return_value.chat.complete.return_value = response

        task = {"task_id": "t1", "prompt": "What is 2+2?"}
        result = agent_with_mock_mistral.run(task)
        assert result == "four"

    @pytest.mark.usefixtures("clear_registries")
    def test_shutdown_clears_client(self, agent_with_mock_mistral):
        agent_with_mock_mistral.shutdown()
        assert agent_with_mock_mistral._adapter._client is None

    @pytest.mark.usefixtures("clear_registries")
    def test_metadata_returns_required_keys(self, agent_with_mock_mistral):
        meta = agent_with_mock_mistral.metadata()
        assert meta["name"] == "MistralAgent"
        assert meta["provider"] == "mistral"
        assert "model" in meta

    @pytest.mark.usefixtures("clear_registries")
    def test_reset_clears_logs(self, agent_with_mock_mistral):
        agent_with_mock_mistral._adapter._request_logs.append({"event": "request"})
        agent_with_mock_mistral._adapter._response_logs.append({"event": "response"})
        agent_with_mock_mistral.reset()
        assert agent_with_mock_mistral._adapter._request_logs == []
        assert agent_with_mock_mistral._adapter._response_logs == []

    @pytest.mark.usefixtures("clear_registries")
    def test_run_uses_model_from_config_llm(self, mistralai_mod, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        cfg = _make_config(llm="mistral-small-2407")
        from llm_reliability.agents.mistral_agent import MistralAgent

        response = _make_mistral_completion(text="hi")
        client_mock = mistralai_mod.Mistral.return_value
        client_mock.chat.complete.return_value = response

        agent = MistralAgent(cfg)
        agent.initialize()
        agent.run({"task_id": "t1", "prompt": "hi"})

        call_kwargs = client_mock.chat.complete.call_args[1]
        assert call_kwargs["model"] == "mistral-small-2407"
