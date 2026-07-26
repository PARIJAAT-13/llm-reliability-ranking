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
        experiment_name="cohere_test",
        benchmark="AgentBoard",
        agent="CohereAgent",
        llm="command-r-plus",
        prompt_version="v1",
        dataset_version="1.0",
        seed=42,
        repetitions=1,
    )
    defaults.update(overrides)
    return Configuration(**defaults)


def _make_cohere_response(text: str = "answer") -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 5

    response = MagicMock()
    response.text = text
    response.token_count = usage
    return response


def _stub_cohere_module() -> ModuleType:
    cohere_mock = ModuleType("cohere")

    class _ApiError(Exception):
        pass

    cohere_mock.UnauthorizedError = type("UnauthorizedError", (_ApiError,), {})
    cohere_mock.TooManyRequestsError = type("TooManyRequestsError", (_ApiError,), {})

    cohere_mock.core = ModuleType("cohere.core")
    cohere_mock.core.api_error = ModuleType("cohere.core.api_error")
    cohere_mock.core.api_error.ApiError = _ApiError

    client_mock = MagicMock()
    cohere_mock.Client = MagicMock(return_value=client_mock)
    return cohere_mock


@pytest.fixture
def cohere_mod() -> Generator[ModuleType, None, None]:
    stub = _stub_cohere_module()
    with patch.dict(
        sys.modules,
        {"cohere": stub, "cohere.core": stub.core, "cohere.core.api_error": stub.core.api_error},
    ):
        yield stub


@pytest.fixture
def config() -> Configuration:
    return _make_config()


@pytest.fixture
def clear_registries():
    from llm_reliability.agents.adapters.provider_registry import ProviderRegistry
    from llm_reliability.runtime.registry import RuntimeRegistry

    ProviderRegistry._adapters.clear()
    RuntimeRegistry._runtimes.clear()
    RuntimeRegistry._initialised = False
    RuntimeRegistry._discovered_module_names.clear()


@pytest.fixture
def agent_with_mock_cohere(config, cohere_mod, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    from llm_reliability.agents.cohere_agent import CohereAgent

    agent = CohereAgent(config)
    agent.initialize()
    return agent


class TestCohereAgent:
    @pytest.mark.usefixtures("clear_registries")
    def test_initialize_raises_without_api_key(self, config, cohere_mod, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        from llm_reliability.agents.cohere_agent import CohereAgent

        agent = CohereAgent(config)
        with pytest.raises(AuthenticationError, match="COHERE_API_KEY"):
            agent.initialize()

    @pytest.mark.usefixtures("clear_registries")
    def test_initialize_succeeds_with_api_key(self, config, cohere_mod, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        from llm_reliability.agents.cohere_agent import CohereAgent

        agent = CohereAgent(config)
        agent.initialize()
        assert agent._adapter._client is not None

    @pytest.mark.usefixtures("clear_registries")
    def test_run_returns_text(self, agent_with_mock_cohere, cohere_mod):
        response = _make_cohere_response(text="four")
        cohere_mod.Client.return_value.chat.return_value = response

        task = {"task_id": "t1", "prompt": "What is 2+2?"}
        result = agent_with_mock_cohere.run(task)
        assert result == "four"

    @pytest.mark.usefixtures("clear_registries")
    def test_shutdown_clears_client(self, agent_with_mock_cohere):
        agent_with_mock_cohere.shutdown()
        assert agent_with_mock_cohere._adapter._client is None

    @pytest.mark.usefixtures("clear_registries")
    def test_metadata_returns_required_keys(self, agent_with_mock_cohere):
        meta = agent_with_mock_cohere.metadata()
        assert meta["name"] == "CohereAgent"
        assert meta["provider"] == "cohere"
        assert "model" in meta

    @pytest.mark.usefixtures("clear_registries")
    def test_reset_clears_logs(self, agent_with_mock_cohere):
        agent_with_mock_cohere._adapter._request_logs.append({"event": "request"})
        agent_with_mock_cohere._adapter._response_logs.append({"event": "response"})
        agent_with_mock_cohere.reset()
        assert agent_with_mock_cohere._adapter._request_logs == []
        assert agent_with_mock_cohere._adapter._response_logs == []

    @pytest.mark.usefixtures("clear_registries")
    def test_run_uses_model_from_config_llm(self, cohere_mod, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        cfg = _make_config(llm="command-r")
        from llm_reliability.agents.cohere_agent import CohereAgent

        response = _make_cohere_response(text="hi")
        client_mock = cohere_mod.Client.return_value
        client_mock.chat.return_value = response

        agent = CohereAgent(cfg)
        agent.initialize()
        agent.run({"task_id": "t1", "prompt": "hi"})

        call_kwargs = client_mock.chat.call_args[1]
        assert call_kwargs["model"] == "command-r"
