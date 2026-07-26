from __future__ import annotations

import os
import sys
from collections.abc import Generator
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_reliability.agents.adapters.exceptions import AuthenticationError
from llm_reliability.configs.config import Configuration


def _make_config(
    agent: str = "AzureOpenAIAgent", llm: str = "gpt-4o", **overrides: Any
) -> Configuration:
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
    openai_mock.AzureOpenAI = MagicMock()
    openai_mock.OpenAI = MagicMock()
    return openai_mock


def _stub_boto3_module() -> ModuleType:
    boto3_mock = ModuleType("boto3")

    def fake_client(service: str, **kwargs: Any) -> MagicMock:
        client = MagicMock()
        client.invoke_model.return_value = {
            "body": MagicMock(
                read=MagicMock(
                    return_value='{"content": [{"type": "text", "text": "bedrock response"}], "stop_reason": "stop", "usage": {"input_tokens": 10, "output_tokens": 5}}'
                )
            )
        }
        return client

    boto3_mock.client = fake_client
    return boto3_mock


def _make_module(name: str) -> ModuleType:
    return ModuleType(name)


def _stub_vertexai_modules() -> dict[str, ModuleType]:
    google = _make_module("google")
    google_cloud = _make_module("google.cloud")
    aiplatform = _make_module("google.cloud.aiplatform")
    aiplatform.init = MagicMock()

    vertexai_pkg = _make_module("vertexai")
    vertexai_preview = _make_module("vertexai.preview")
    vertexai_gen = _make_module("vertexai.preview.generative_models")
    gen_model_cls = MagicMock()
    gen_model_instance = MagicMock()
    gen_model_instance.generate_content.return_value = MagicMock(
        text="vertex response",
        usage_metadata=MagicMock(prompt_token_count=10, candidates_token_count=5),
    )
    gen_model_cls.return_value = gen_model_instance
    vertexai_gen.GenerativeModel = gen_model_cls

    return {
        "google": google,
        "google.cloud": google_cloud,
        "google.cloud.aiplatform": aiplatform,
        "vertexai": vertexai_pkg,
        "vertexai.preview": vertexai_preview,
        "vertexai.preview.generative_models": vertexai_gen,
    }


def _stub_litellm_module() -> ModuleType:
    litellm = ModuleType("litellm")

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5

    message = MagicMock()
    message.content = "litellm response"

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = "test-litellm-model"

    litellm.completion = MagicMock(return_value=response)
    return litellm


class BaseAgentTestSetup:
    @pytest.fixture(autouse=True)
    def clear_registries(self):
        from llm_reliability.agents.adapters.provider_registry import \
            ProviderRegistry
        from llm_reliability.runtime.registry import RuntimeRegistry

        ProviderRegistry._adapters.clear()
        RuntimeRegistry._runtimes.clear()
        RuntimeRegistry._initialised = False
        RuntimeRegistry._discovered_module_names.clear()


# --------------------------------------------------------------------------- #
# AzureOpenAIAgent
# --------------------------------------------------------------------------- #


class TestAzureOpenAIAgent(BaseAgentTestSetup):
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")

    def test_initialize_succeeds_with_env_vars(self):
        stub = _stub_openai_module()
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.azure_openai_agent import \
                AzureOpenAIAgent

            agent = AzureOpenAIAgent(_make_config(agent="AzureOpenAIAgent"))
            agent.initialize()
            assert agent._adapter._client is not None

    def test_initialize_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        stub = _stub_openai_module()
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.azure_openai_agent import \
                AzureOpenAIAgent

            agent = AzureOpenAIAgent(_make_config(agent="AzureOpenAIAgent"))
            with pytest.raises(AuthenticationError):
                agent.initialize()

    def test_initialize_raises_without_endpoint(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        stub = _stub_openai_module()
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.azure_openai_agent import \
                AzureOpenAIAgent

            agent = AzureOpenAIAgent(_make_config(agent="AzureOpenAIAgent"))
            with pytest.raises(AuthenticationError):
                agent.initialize()

    def test_run_returns_text(self):
        stub = _stub_openai_module()
        completion = _make_openai_completion(text="azure answer")
        client_mock = stub.AzureOpenAI.return_value
        client_mock.chat.completions.create.return_value = completion
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.azure_openai_agent import \
                AzureOpenAIAgent

            agent = AzureOpenAIAgent(_make_config(agent="AzureOpenAIAgent"))
            agent.initialize()
            result = agent.run({"task_id": "t1", "prompt": "What is 2+2?"})
            assert result == "azure answer"

    def test_metadata(self):
        stub = _stub_openai_module()
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.azure_openai_agent import \
                AzureOpenAIAgent

            agent = AzureOpenAIAgent(_make_config(agent="AzureOpenAIAgent"))
            agent.initialize()
            meta = agent.metadata()
            assert meta["name"] == "AzureOpenAIAgent"
            assert meta["provider"] == "azure_openai"

    def test_shutdown_clears_client(self):
        stub = _stub_openai_module()
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.azure_openai_agent import \
                AzureOpenAIAgent

            agent = AzureOpenAIAgent(_make_config(agent="AzureOpenAIAgent"))
            agent.initialize()
            agent.shutdown()
            assert agent._adapter._client is None

    def test_lifecycle(self):
        stub = _stub_openai_module()
        completion = _make_openai_completion(text="lifecycle")
        client_mock = stub.AzureOpenAI.return_value
        client_mock.chat.completions.create.return_value = completion
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.azure_openai_agent import \
                AzureOpenAIAgent

            agent = AzureOpenAIAgent(_make_config(agent="AzureOpenAIAgent"))
            agent.initialize()
            result = agent.run({"prompt": "test"})
            assert result == "lifecycle"
            meta = agent.metadata()
            assert meta["name"] == "AzureOpenAIAgent"
            agent.shutdown()
            assert agent._adapter._client is None


# --------------------------------------------------------------------------- #
# BedrockAgent
# --------------------------------------------------------------------------- #


class TestBedrockAgent(BaseAgentTestSetup):
    def test_initialize_raises_without_creds(self, monkeypatch):
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        stub = _stub_boto3_module()
        with patch.dict(sys.modules, {"boto3": stub}):
            from llm_reliability.agents.bedrock_agent import BedrockAgent

            agent = BedrockAgent(
                _make_config(agent="BedrockAgent", llm="anthropic.claude-3-5-sonnet-20241022-v2:0")
            )
            agent.initialize()
            assert agent._adapter._client is not None

    def test_run_returns_text(self):
        stub = _stub_boto3_module()
        with patch.dict(sys.modules, {"boto3": stub}):
            from llm_reliability.agents.bedrock_agent import BedrockAgent

            agent = BedrockAgent(_make_config(agent="BedrockAgent"))
            agent.initialize()
            result = agent.run({"task_id": "t1", "prompt": "hello"})
            assert result == "bedrock response"

    def test_metadata(self):
        stub = _stub_boto3_module()
        with patch.dict(sys.modules, {"boto3": stub}):
            from llm_reliability.agents.bedrock_agent import BedrockAgent

            agent = BedrockAgent(_make_config(agent="BedrockAgent"))
            agent.initialize()
            meta = agent.metadata()
            assert meta["name"] == "BedrockAgent"
            assert meta["provider"] == "bedrock"

    def test_shutdown_clears_client(self):
        stub = _stub_boto3_module()
        with patch.dict(sys.modules, {"boto3": stub}):
            from llm_reliability.agents.bedrock_agent import BedrockAgent

            agent = BedrockAgent(_make_config(agent="BedrockAgent"))
            agent.initialize()
            agent.shutdown()
            assert agent._adapter._client is None

    def test_lifecycle(self):
        stub = _stub_boto3_module()
        with patch.dict(sys.modules, {"boto3": stub}):
            from llm_reliability.agents.bedrock_agent import BedrockAgent

            agent = BedrockAgent(_make_config(agent="BedrockAgent"))
            agent.initialize()
            result = agent.run({"prompt": "test"})
            assert result == "bedrock response"
            meta = agent.metadata()
            assert meta["name"] == "BedrockAgent"
            agent.shutdown()
            assert agent._adapter._client is None


# --------------------------------------------------------------------------- #
# VertexAgent
# --------------------------------------------------------------------------- #


class TestVertexAgent(BaseAgentTestSetup):
    @pytest.fixture(autouse=True)
    def _vertex_modules(self):
        modules = _stub_vertexai_modules()
        with patch.dict(sys.modules, modules):
            yield modules

    def test_initialize_raises_without_project(self, monkeypatch):
        monkeypatch.delenv("VERTEX_AI_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        from llm_reliability.agents.vertex_agent import VertexAgent

        agent = VertexAgent(_make_config(agent="VertexAgent"))
        with pytest.raises(AuthenticationError):
            agent.initialize()

    def test_initialize_succeeds_with_project(self, monkeypatch):
        monkeypatch.setenv("VERTEX_AI_PROJECT", "test-project")
        from llm_reliability.agents.vertex_agent import VertexAgent

        agent = VertexAgent(_make_config(agent="VertexAgent"))
        agent.initialize()
        assert agent._adapter._client is not None

    def test_run_returns_text(self, monkeypatch):
        monkeypatch.setenv("VERTEX_AI_PROJECT", "test-project")
        from llm_reliability.agents.vertex_agent import VertexAgent

        agent = VertexAgent(_make_config(agent="VertexAgent"))
        agent.initialize()
        result = agent.run({"task_id": "t1", "prompt": "hello"})
        assert result == "vertex response"

    def test_metadata(self, monkeypatch):
        monkeypatch.setenv("VERTEX_AI_PROJECT", "test-project")
        from llm_reliability.agents.vertex_agent import VertexAgent

        agent = VertexAgent(_make_config(agent="VertexAgent"))
        agent.initialize()
        meta = agent.metadata()
        assert meta["name"] == "VertexAgent"
        assert meta["provider"] == "vertex"

    def test_shutdown_clears_client(self, monkeypatch):
        monkeypatch.setenv("VERTEX_AI_PROJECT", "test-project")
        from llm_reliability.agents.vertex_agent import VertexAgent

        agent = VertexAgent(_make_config(agent="VertexAgent"))
        agent.initialize()
        agent.shutdown()
        assert agent._adapter._client is None

    def test_lifecycle(self, monkeypatch):
        monkeypatch.setenv("VERTEX_AI_PROJECT", "test-project")
        from llm_reliability.agents.vertex_agent import VertexAgent

        agent = VertexAgent(_make_config(agent="VertexAgent"))
        agent.initialize()
        result = agent.run({"prompt": "test"})
        assert result == "vertex response"
        meta = agent.metadata()
        assert meta["name"] == "VertexAgent"
        agent.shutdown()
        assert agent._adapter._client is None

    def test_initialise_uses_google_cloud_project_fallback(self, monkeypatch):
        monkeypatch.delenv("VERTEX_AI_PROJECT", raising=False)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fallback-project")
        from llm_reliability.agents.vertex_agent import VertexAgent

        agent = VertexAgent(_make_config(agent="VertexAgent"))
        agent.initialize()
        assert agent._adapter._client is not None


# --------------------------------------------------------------------------- #
# FireworksAgent
# --------------------------------------------------------------------------- #


class TestFireworksAgent(BaseAgentTestSetup):
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-fw-key")

    def test_initialize_succeeds(self):
        stub = _stub_openai_module()
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.fireworks_agent import FireworksAgent

            agent = FireworksAgent(_make_config(agent="FireworksAgent"))
            agent.initialize()
            assert agent._adapter._client is not None

    def test_initialize_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        stub = _stub_openai_module()
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.fireworks_agent import FireworksAgent

            agent = FireworksAgent(_make_config(agent="FireworksAgent"))
            with pytest.raises(AuthenticationError):
                agent.initialize()

    def test_run_returns_text(self):
        stub = _stub_openai_module()
        completion = _make_openai_completion(text="fireworks answer")
        stub.OpenAI.return_value.chat.completions.create.return_value = completion
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.fireworks_agent import FireworksAgent

            agent = FireworksAgent(_make_config(agent="FireworksAgent"))
            agent.initialize()
            result = agent.run({"prompt": "test"})
            assert result == "fireworks answer"

    def test_metadata(self):
        stub = _stub_openai_module()
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.fireworks_agent import FireworksAgent

            agent = FireworksAgent(_make_config(agent="FireworksAgent"))
            agent.initialize()
            meta = agent.metadata()
            assert meta["name"] == "FireworksAgent"
            assert meta["provider"] == "fireworks"

    def test_shutdown_clears_client(self):
        stub = _stub_openai_module()
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.fireworks_agent import FireworksAgent

            agent = FireworksAgent(_make_config(agent="FireworksAgent"))
            agent.initialize()
            agent.shutdown()
            assert agent._adapter._client is None

    def test_lifecycle(self):
        stub = _stub_openai_module()
        completion = _make_openai_completion(text="fw lifecycle")
        stub.OpenAI.return_value.chat.completions.create.return_value = completion
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.fireworks_agent import FireworksAgent

            agent = FireworksAgent(_make_config(agent="FireworksAgent"))
            agent.initialize()
            result = agent.run({"prompt": "hello"})
            assert result == "fw lifecycle"
            agent.shutdown()
            assert agent._adapter._client is None


# --------------------------------------------------------------------------- #
# TogetherAgent
# --------------------------------------------------------------------------- #


class TestTogetherAgent(BaseAgentTestSetup):
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setenv("TOGETHER_API_KEY", "test-together-key")

    def test_initialize_succeeds(self):
        stub = _stub_openai_module()
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.together_agent import TogetherAgent

            agent = TogetherAgent(_make_config(agent="TogetherAgent"))
            agent.initialize()
            assert agent._adapter._client is not None

    def test_initialize_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        stub = _stub_openai_module()
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.together_agent import TogetherAgent

            agent = TogetherAgent(_make_config(agent="TogetherAgent"))
            with pytest.raises(AuthenticationError):
                agent.initialize()

    def test_run_returns_text(self):
        stub = _stub_openai_module()
        completion = _make_openai_completion(text="together answer")
        stub.OpenAI.return_value.chat.completions.create.return_value = completion
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.together_agent import TogetherAgent

            agent = TogetherAgent(_make_config(agent="TogetherAgent"))
            agent.initialize()
            result = agent.run({"prompt": "test"})
            assert result == "together answer"

    def test_metadata(self):
        stub = _stub_openai_module()
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.together_agent import TogetherAgent

            agent = TogetherAgent(_make_config(agent="TogetherAgent"))
            agent.initialize()
            meta = agent.metadata()
            assert meta["name"] == "TogetherAgent"
            assert meta["provider"] == "together"

    def test_shutdown_clears_client(self):
        stub = _stub_openai_module()
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.together_agent import TogetherAgent

            agent = TogetherAgent(_make_config(agent="TogetherAgent"))
            agent.initialize()
            agent.shutdown()
            assert agent._adapter._client is None

    def test_run_passes_seed(self):
        stub = _stub_openai_module()
        completion = _make_openai_completion(text="ok")
        client_mock = stub.OpenAI.return_value
        client_mock.chat.completions.create.return_value = completion
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.together_agent import TogetherAgent

            agent = TogetherAgent(_make_config(agent="TogetherAgent", seed=42))
            agent.initialize()
            agent.run({"prompt": "hello"})
            call_kwargs = client_mock.chat.completions.create.call_args[1]
            assert call_kwargs["seed"] == 42

    def test_lifecycle(self):
        stub = _stub_openai_module()
        completion = _make_openai_completion(text="tg lifecycle")
        stub.OpenAI.return_value.chat.completions.create.return_value = completion
        with patch.dict(sys.modules, {"openai": stub}):
            from llm_reliability.agents.together_agent import TogetherAgent

            agent = TogetherAgent(_make_config(agent="TogetherAgent"))
            agent.initialize()
            result = agent.run({"prompt": "test"})
            assert result == "tg lifecycle"
            agent.shutdown()
            assert agent._adapter._client is None


# --------------------------------------------------------------------------- #
# LiteLLMAgent
# --------------------------------------------------------------------------- #


class TestLiteLLMAgent(BaseAgentTestSetup):
    def test_initialize_succeeds(self):
        stub = _stub_litellm_module()
        with patch.dict(sys.modules, {"litellm": stub}):
            from llm_reliability.agents.litellm_agent import LiteLLMAgent

            agent = LiteLLMAgent(_make_config(agent="LiteLLMAgent"))
            agent.initialize()
            assert agent._adapter.health_check()

    def test_initialize_does_not_require_env_vars(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        stub = _stub_litellm_module()
        with patch.dict(sys.modules, {"litellm": stub}):
            from llm_reliability.agents.litellm_agent import LiteLLMAgent

            agent = LiteLLMAgent(_make_config(agent="LiteLLMAgent"))
            agent.initialize()
            assert agent._adapter.health_check()

    def test_run_returns_text(self):
        stub = _stub_litellm_module()
        with patch.dict(sys.modules, {"litellm": stub}):
            from llm_reliability.agents.litellm_agent import LiteLLMAgent

            agent = LiteLLMAgent(_make_config(agent="LiteLLMAgent"))
            agent.initialize()
            result = agent.run({"prompt": "test"})
            assert result == "litellm response"

    def test_metadata(self):
        stub = _stub_litellm_module()
        with patch.dict(sys.modules, {"litellm": stub}):
            from llm_reliability.agents.litellm_agent import LiteLLMAgent

            agent = LiteLLMAgent(_make_config(agent="LiteLLMAgent"))
            agent.initialize()
            meta = agent.metadata()
            assert meta["name"] == "LiteLLMAgent"
            assert meta["provider"] == "litellm"

    def test_shutdown(self):
        stub = _stub_litellm_module()
        with patch.dict(sys.modules, {"litellm": stub}):
            from llm_reliability.agents.litellm_agent import LiteLLMAgent

            agent = LiteLLMAgent(_make_config(agent="LiteLLMAgent"))
            agent.initialize()
            agent.shutdown()
            assert agent._adapter.health_check()

    def test_lifecycle(self):
        stub = _stub_litellm_module()
        with patch.dict(sys.modules, {"litellm": stub}):
            from llm_reliability.agents.litellm_agent import LiteLLMAgent

            agent = LiteLLMAgent(_make_config(agent="LiteLLMAgent"))
            agent.initialize()
            result = agent.run({"prompt": "lifecycle"})
            assert result == "litellm response"
            meta = agent.metadata()
            assert meta["name"] == "LiteLLMAgent"
            agent.shutdown()
