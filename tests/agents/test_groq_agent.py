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
        experiment_name="groq_test",
        benchmark="AgentBoard",
        agent="GroqAgent",
        llm="llama-3.3-70b-versatile",
        prompt_version="v1",
        dataset_version="1.0",
        seed=42,
        repetitions=1,
    )
    defaults.update(overrides)
    return Configuration(**defaults)


def _make_groq_completion(text: str = "answer", finish_reason: str = "stop") -> MagicMock:
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
    completion.model = "llama-3.3-70b-versatile"
    completion.id = "chatcmpl-test123"
    return completion


def _stub_groq_module() -> ModuleType:
    groq_mock = ModuleType("groq")

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

    groq_mock.APIError = _APIError
    groq_mock.AuthenticationError = _AuthenticationError
    groq_mock.RateLimitError = _RateLimitError
    groq_mock.APIConnectionError = _APIConnectionError
    groq_mock.APITimeoutError = _APITimeoutError

    client_mock = MagicMock()
    groq_mock.Groq = MagicMock(return_value=client_mock)
    return groq_mock


@pytest.fixture
def groq_mod() -> Generator[ModuleType, None, None]:
    stub = _stub_groq_module()
    with patch.dict(sys.modules, {"groq": stub}):
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
def agent_with_mock_groq(config, groq_mod, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-key")
    from llm_reliability.agents.groq_agent import GroqAgent

    agent = GroqAgent(config)
    agent.initialize()
    return agent


class TestGroqAgent:
    @pytest.mark.usefixtures("clear_registries")
    def test_initialize_raises_without_api_key(self, config, groq_mod, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from llm_reliability.agents.groq_agent import GroqAgent

        agent = GroqAgent(config)
        with pytest.raises(AuthenticationError, match="GROQ_API_KEY"):
            agent.initialize()

    @pytest.mark.usefixtures("clear_registries")
    def test_initialize_succeeds_with_api_key(self, config, groq_mod, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        from llm_reliability.agents.groq_agent import GroqAgent

        agent = GroqAgent(config)
        agent.initialize()
        assert agent._adapter._client is not None

    @pytest.mark.usefixtures("clear_registries")
    def test_run_returns_text(self, agent_with_mock_groq, groq_mod):
        completion = _make_groq_completion(text="four")
        groq_mod.Groq.return_value.chat.completions.create.return_value = completion

        task = {"task_id": "t1", "prompt": "What is 2+2?"}
        result = agent_with_mock_groq.run(task)
        assert result == "four"

    @pytest.mark.usefixtures("clear_registries")
    def test_run_does_not_pass_seed(self, config, groq_mod, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        from llm_reliability.agents.groq_agent import GroqAgent

        completion = _make_groq_completion(text="ok")
        client_mock = groq_mod.Groq.return_value
        client_mock.chat.completions.create.return_value = completion

        agent = GroqAgent(config)
        agent.initialize()
        agent.run({"task_id": "t1", "prompt": "hello"})

        call_kwargs = client_mock.chat.completions.create.call_args[1]
        assert "seed" not in call_kwargs

    @pytest.mark.usefixtures("clear_registries")
    def test_shutdown_closes_client(self, agent_with_mock_groq):
        agent_with_mock_groq.shutdown()
        assert agent_with_mock_groq._adapter._client is None

    @pytest.mark.usefixtures("clear_registries")
    def test_metadata_returns_required_keys(self, agent_with_mock_groq):
        meta = agent_with_mock_groq.metadata()
        assert meta["name"] == "GroqAgent"
        assert meta["provider"] == "groq"
        assert "model" in meta

    @pytest.mark.usefixtures("clear_registries")
    def test_reset_clears_logs(self, agent_with_mock_groq):
        agent_with_mock_groq._adapter._request_logs.append({"event": "request"})
        agent_with_mock_groq._adapter._response_logs.append({"event": "response"})
        agent_with_mock_groq.reset()
        assert agent_with_mock_groq._adapter._request_logs == []
        assert agent_with_mock_groq._adapter._response_logs == []

    @pytest.mark.usefixtures("clear_registries")
    def test_run_uses_model_from_config_llm(self, groq_mod, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        cfg = _make_config(llm="llama-3.1-8b")
        from llm_reliability.agents.groq_agent import GroqAgent

        completion = _make_groq_completion(text="hi")
        client_mock = groq_mod.Groq.return_value
        client_mock.chat.completions.create.return_value = completion

        agent = GroqAgent(cfg)
        agent.initialize()
        agent.run({"task_id": "t1", "prompt": "hi"})

        call_kwargs = client_mock.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "llama-3.1-8b"
