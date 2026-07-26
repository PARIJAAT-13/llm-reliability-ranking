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
        experiment_name="openrouter_test",
        benchmark="AgentBoard",
        agent="OpenRouterAgent",
        llm="openai/gpt-4o",
        prompt_version="v1",
        dataset_version="1.0",
        seed=42,
        repetitions=1,
    )
    defaults.update(overrides)
    return Configuration(**defaults)


def _make_openai_completion(text: str = "answer", finish_reason: str = "stop") -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5

    message = MagicMock()
    message.content = text

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = usage
    completion.model = "openai/gpt-4o"
    completion.id = "chatcmpl-test123"
    return completion


def _stub_openai_module() -> ModuleType:
    openai_mock = ModuleType("openai")

    class _APIError(Exception):
        pass

    class _AuthenticationError(_APIError):
        pass

    class _RateLimitError(_APIError):
        pass

    class _APIConnectionError(_APIError):
        pass

    class _APITimeoutError(_APIConnectionError):
        pass

    class _BadRequestError(_APIError):
        pass

    openai_mock.APIError = _APIError
    openai_mock.AuthenticationError = _AuthenticationError
    openai_mock.RateLimitError = _RateLimitError
    openai_mock.APIConnectionError = _APIConnectionError
    openai_mock.APITimeoutError = _APITimeoutError
    openai_mock.BadRequestError = _BadRequestError

    client_mock = MagicMock()
    openai_mock.OpenAI = MagicMock(return_value=client_mock)
    return openai_mock


@pytest.fixture
def openai_mod() -> Generator[ModuleType, None, None]:
    stub = _stub_openai_module()
    with patch.dict(sys.modules, {"openai": stub}):
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
def agent_with_mock_openai(config, openai_mod, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    from llm_reliability.agents.openrouter_agent import OpenRouterAgent

    agent = OpenRouterAgent(config)
    agent.initialize()
    return agent


class TestOpenRouterAgent:
    @pytest.mark.usefixtures("clear_registries")
    def test_initialize_raises_without_api_key(self, config, openai_mod, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from llm_reliability.agents.openrouter_agent import OpenRouterAgent

        agent = OpenRouterAgent(config)
        with pytest.raises(AuthenticationError, match="OPENROUTER_API_KEY"):
            agent.initialize()

    @pytest.mark.usefixtures("clear_registries")
    def test_initialize_succeeds_with_api_key(self, config, openai_mod, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        from llm_reliability.agents.openrouter_agent import OpenRouterAgent

        agent = OpenRouterAgent(config)
        agent.initialize()
        assert agent._adapter._client is not None

    @pytest.mark.usefixtures("clear_registries")
    def test_run_returns_text(self, agent_with_mock_openai, openai_mod):
        completion = _make_openai_completion(text="four")
        openai_mod.OpenAI.return_value.chat.completions.create.return_value = completion

        task = {"task_id": "t1", "prompt": "What is 2+2?"}
        result = agent_with_mock_openai.run(task)
        assert result == "four"

    @pytest.mark.usefixtures("clear_registries")
    def test_run_passes_seed_to_request(self, config, openai_mod, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        from llm_reliability.agents.openrouter_agent import OpenRouterAgent

        completion = _make_openai_completion(text="ok")
        client_mock = openai_mod.OpenAI.return_value
        client_mock.chat.completions.create.return_value = completion

        agent = OpenRouterAgent(config)
        agent.initialize()
        agent.run({"task_id": "t1", "prompt": "hello"})

        call_kwargs = client_mock.chat.completions.create.call_args[1]
        assert call_kwargs.get("seed") == config.seed

    @pytest.mark.usefixtures("clear_registries")
    def test_shutdown_closes_client(self, agent_with_mock_openai):
        agent_with_mock_openai.shutdown()
        assert agent_with_mock_openai._adapter._client is None

    @pytest.mark.usefixtures("clear_registries")
    def test_metadata_returns_required_keys(self, agent_with_mock_openai):
        meta = agent_with_mock_openai.metadata()
        assert meta["name"] == "OpenRouterAgent"
        assert meta["provider"] == "openrouter"
        assert "model" in meta

    @pytest.mark.usefixtures("clear_registries")
    def test_reset_clears_logs(self, agent_with_mock_openai):
        agent_with_mock_openai._adapter._request_logs.append({"event": "request"})
        agent_with_mock_openai._adapter._response_logs.append({"event": "response"})
        agent_with_mock_openai.reset()
        assert agent_with_mock_openai._adapter._request_logs == []
        assert agent_with_mock_openai._adapter._response_logs == []

    @pytest.mark.usefixtures("clear_registries")
    def test_run_uses_model_from_config_llm(self, openai_mod, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        cfg = _make_config(llm="openai/gpt-3.5-turbo")
        from llm_reliability.agents.openrouter_agent import OpenRouterAgent

        completion = _make_openai_completion(text="hi")
        client_mock = openai_mod.OpenAI.return_value
        client_mock.chat.completions.create.return_value = completion

        agent = OpenRouterAgent(cfg)
        agent.initialize()
        agent.run({"task_id": "t1", "prompt": "hi"})

        call_kwargs = client_mock.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "openai/gpt-3.5-turbo"
