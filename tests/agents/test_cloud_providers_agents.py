from __future__ import annotations

import sys
from collections.abc import Generator
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_reliability.agents.adapters.exceptions import AuthenticationError
from llm_reliability.configs.config import Configuration


def _make_config(agent: str = "XAIAgent", llm: str = "grok-2", **overrides: Any) -> Configuration:
    defaults: dict[str, Any] = dict(
        experiment_name=f"{agent}_test",
        benchmark="AgentBoard",
        agent=agent,
        llm=llm,
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
    completion.model = "test-model"
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

    openai_mock.APIError = _APIError
    openai_mock.AuthenticationError = _AuthenticationError
    openai_mock.RateLimitError = _RateLimitError
    openai_mock.APIConnectionError = _APIConnectionError
    openai_mock.APITimeoutError = _APITimeoutError

    client_mock = MagicMock()
    openai_mock.OpenAI = MagicMock(return_value=client_mock)
    return openai_mock


@pytest.fixture
def openai_mod() -> Generator[ModuleType, None, None]:
    stub = _stub_openai_module()
    with patch.dict(sys.modules, {"openai": stub}):
        yield stub


@pytest.fixture
def clear_registries():
    from llm_reliability.agents.adapters.provider_registry import \
        ProviderRegistry
    from llm_reliability.runtime.registry import RuntimeRegistry

    ProviderRegistry._adapters.clear()
    RuntimeRegistry._runtimes.clear()
    RuntimeRegistry._initialised = False
    RuntimeRegistry._discovered_module_names.clear()


XAI_CFG = ("xai", "XAIAgent", "XAI_API_KEY", "grok-2")
PERPLEXITY_CFG = ("perplexity", "PerplexityAgent", "PERPLEXITY_API_KEY", "sonar-pro")
SAMBANOVA_CFG = ("sambanova", "SambaNovaAgent", "SAMBANOVA_API_KEY", "Meta-Llama-3.3-70B-Instruct")
CEREBRAS_CFG = ("cerebras", "CerebrasAgent", "CEREBRAS_API_KEY", "llama3.1-8b")
NIM_CFG = ("nvidia_nim", "NIMAgent", "NVIDIA_API_KEY", "meta/llama-3.3-70b-instruct")
SGLANG_CFG = ("sglang", "SGLangAgent", "SGLANG_API_KEY", "meta-llama/Llama-3.3-70B-Instruct")

AGENTS = [XAI_CFG, PERPLEXITY_CFG, SAMBANOVA_CFG, CEREBRAS_CFG, NIM_CFG, SGLANG_CFG]
AGENTS_WITH_AUTH = [XAI_CFG, PERPLEXITY_CFG, SAMBANOVA_CFG, CEREBRAS_CFG, NIM_CFG]


class TestCloudProvidersAgents:
    @pytest.mark.usefixtures("clear_registries")
    @pytest.mark.parametrize("provider,name,env_var,default_model", AGENTS_WITH_AUTH)
    def test_initialize_raises_without_api_key(
        self, provider, name, env_var, default_model, openai_mod, monkeypatch, request
    ):
        monkeypatch.delenv(env_var, raising=False)
        if provider == "nvidia_nim":
            monkeypatch.delenv("NGC_API_KEY", raising=False)
        module_path = (
            f"llm_reliability.agents.{provider}_agent"
            if provider != "nvidia_nim"
            else "llm_reliability.agents.nim_agent"
        )
        agent_cls = getattr(__import__(module_path, fromlist=[name]), name)
        agent = agent_cls(_make_config(agent=name, llm=default_model))
        with pytest.raises(AuthenticationError):
            agent.initialize()

    @pytest.mark.usefixtures("clear_registries")
    def test_sglang_initializes_without_env_var(self, openai_mod, monkeypatch):
        monkeypatch.delenv("SGLANG_API_KEY", raising=False)
        from llm_reliability.agents.sglang_agent import SGLangAgent

        agent = SGLangAgent(
            _make_config(agent="SGLangAgent", llm="meta-llama/Llama-3.3-70B-Instruct")
        )
        agent.initialize()
        assert agent._adapter._client is not None

    @pytest.mark.usefixtures("clear_registries")
    @pytest.mark.parametrize("provider,name,env_var,default_model", AGENTS)
    def test_initialize_succeeds_with_api_key(
        self, provider, name, env_var, default_model, openai_mod, monkeypatch
    ):
        monkeypatch.setenv(env_var, "test-key")
        module_path = (
            f"llm_reliability.agents.{provider}_agent"
            if provider != "nvidia_nim"
            else "llm_reliability.agents.nim_agent"
        )
        agent_cls = getattr(__import__(module_path, fromlist=[name]), name)
        agent = agent_cls(_make_config(agent=name, llm=default_model))
        agent.initialize()
        assert agent._adapter._client is not None

    @pytest.mark.usefixtures("clear_registries")
    @pytest.mark.parametrize("provider,name,env_var,default_model", AGENTS)
    def test_run_returns_text(
        self, provider, name, env_var, default_model, openai_mod, monkeypatch
    ):
        monkeypatch.setenv(env_var, "test-key")
        completion = _make_openai_completion(text="four")
        openai_mod.OpenAI.return_value.chat.completions.create.return_value = completion
        module_path = (
            f"llm_reliability.agents.{provider}_agent"
            if provider != "nvidia_nim"
            else "llm_reliability.agents.nim_agent"
        )
        agent_cls = getattr(__import__(module_path, fromlist=[name]), name)
        agent = agent_cls(_make_config(agent=name, llm=default_model))
        agent.initialize()
        result = agent.run({"task_id": "t1", "prompt": "What is 2+2?"})
        assert result == "four"

    @pytest.mark.usefixtures("clear_registries")
    @pytest.mark.parametrize("provider,name,env_var,default_model", AGENTS)
    def test_run_does_not_pass_seed(
        self, provider, name, env_var, default_model, openai_mod, monkeypatch
    ):
        monkeypatch.setenv(env_var, "test-key")
        completion = _make_openai_completion(text="ok")
        client_mock = openai_mod.OpenAI.return_value
        client_mock.chat.completions.create.return_value = completion
        module_path = (
            f"llm_reliability.agents.{provider}_agent"
            if provider != "nvidia_nim"
            else "llm_reliability.agents.nim_agent"
        )
        agent_cls = getattr(__import__(module_path, fromlist=[name]), name)
        agent = agent_cls(_make_config(agent=name, llm=default_model))
        agent.initialize()
        agent.run({"task_id": "t1", "prompt": "hello"})
        call_kwargs = client_mock.chat.completions.create.call_args[1]
        assert "seed" not in call_kwargs

    @pytest.mark.usefixtures("clear_registries")
    @pytest.mark.parametrize("provider,name,env_var,default_model", AGENTS)
    def test_shutdown_closes_client(
        self, provider, name, env_var, default_model, openai_mod, monkeypatch
    ):
        monkeypatch.setenv(env_var, "test-key")
        module_path = (
            f"llm_reliability.agents.{provider}_agent"
            if provider != "nvidia_nim"
            else "llm_reliability.agents.nim_agent"
        )
        agent_cls = getattr(__import__(module_path, fromlist=[name]), name)
        agent = agent_cls(_make_config(agent=name, llm=default_model))
        agent.initialize()
        agent.shutdown()
        assert agent._adapter._client is None

    @pytest.mark.usefixtures("clear_registries")
    @pytest.mark.parametrize(
        "provider,name,env_var,default_model,expected_provider",
        [
            ("xai", "XAIAgent", "XAI_API_KEY", "grok-2", "xai"),
            ("perplexity", "PerplexityAgent", "PERPLEXITY_API_KEY", "sonar-pro", "perplexity"),
            (
                "sambanova",
                "SambaNovaAgent",
                "SAMBANOVA_API_KEY",
                "Meta-Llama-3.3-70B-Instruct",
                "sambanova",
            ),
            ("cerebras", "CerebrasAgent", "CEREBRAS_API_KEY", "llama3.1-8b", "cerebras"),
            (
                "nvidia_nim",
                "NIMAgent",
                "NVIDIA_API_KEY",
                "meta/llama-3.3-70b-instruct",
                "nvidia_nim",
            ),
            (
                "sglang",
                "SGLangAgent",
                "SGLANG_API_KEY",
                "meta-llama/Llama-3.3-70B-Instruct",
                "sglang",
            ),
        ],
    )
    def test_metadata_returns_required_keys(
        self, provider, name, env_var, default_model, expected_provider, openai_mod, monkeypatch
    ):
        monkeypatch.setenv(env_var, "test-key")
        module_path = (
            f"llm_reliability.agents.{provider}_agent"
            if provider != "nvidia_nim"
            else "llm_reliability.agents.nim_agent"
        )
        agent_cls = getattr(__import__(module_path, fromlist=[name]), name)
        agent = agent_cls(_make_config(agent=name, llm=default_model))
        agent.initialize()
        meta = agent.metadata()
        assert meta["name"] == name
        assert meta["provider"] == expected_provider
        assert "model" in meta

    @pytest.mark.usefixtures("clear_registries")
    @pytest.mark.parametrize("provider,name,env_var,default_model", AGENTS)
    def test_reset_clears_logs(
        self, provider, name, env_var, default_model, openai_mod, monkeypatch
    ):
        monkeypatch.setenv(env_var, "test-key")
        module_path = (
            f"llm_reliability.agents.{provider}_agent"
            if provider != "nvidia_nim"
            else "llm_reliability.agents.nim_agent"
        )
        agent_cls = getattr(__import__(module_path, fromlist=[name]), name)
        agent = agent_cls(_make_config(agent=name, llm=default_model))
        agent.initialize()
        agent._adapter._request_logs.append({"event": "request"})
        agent._adapter._response_logs.append({"event": "response"})
        agent.reset()
        assert agent._adapter._request_logs == []
        assert agent._adapter._response_logs == []

    @pytest.mark.usefixtures("clear_registries")
    @pytest.mark.parametrize("provider,name,env_var,default_model", AGENTS)
    def test_run_uses_model_from_config_llm(
        self, provider, name, env_var, default_model, openai_mod, monkeypatch
    ):
        monkeypatch.setenv(env_var, "test-key")
        custom_model = "custom-model-v1"
        cfg = _make_config(agent=name, llm=custom_model)
        completion = _make_openai_completion(text="hi")
        client_mock = openai_mod.OpenAI.return_value
        client_mock.chat.completions.create.return_value = completion
        module_path = (
            f"llm_reliability.agents.{provider}_agent"
            if provider != "nvidia_nim"
            else "llm_reliability.agents.nim_agent"
        )
        agent_cls = getattr(__import__(module_path, fromlist=[name]), name)
        agent = agent_cls(cfg)
        agent.initialize()
        agent.run({"task_id": "t1", "prompt": "hi"})
        call_kwargs = client_mock.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == custom_model
