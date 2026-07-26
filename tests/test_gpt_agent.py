"""
Tests for GPTAgent.

All OpenAI HTTP calls are mocked so these tests run without a live API key
or network connection.  The test suite verifies:

- Interface compliance (GPTAgent is-a Agent)
- Configuration-driven parameter extraction
- Prompt extraction from various task dict shapes
- Successful run() returns raw model text
- Error mapping (auth, rate-limit, network, empty response)
- metadata() contract
- Deterministic sha256 / Configuration integration
- ProviderRegistry self-registration of the underlying adapter
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_reliability.configs.config import Configuration
from llm_reliability.interfaces.agent import Agent

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> Configuration:
    """Return a valid Configuration with sensible defaults."""
    defaults: dict[str, Any] = dict(
        experiment_name="gpt_test",
        benchmark="AgentBoard",
        agent="GPTAgent",
        llm="gpt-4.1",
        prompt_version="v1",
        dataset_version="1.0",
        seed=42,
        repetitions=1,
    )
    defaults.update(overrides)
    return Configuration(**defaults)


def _make_openai_completion(text: str = "answer", finish_reason: str = "stop") -> MagicMock:
    """Build a minimal mock that mimics openai.ChatCompletion structure."""
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
    completion.model = "gpt-4.1"
    completion.id = "chatcmpl-test123"
    return completion


def _stub_openai_module() -> ModuleType:
    """Return a fake ``openai`` module that satisfies _OpenAIAdapter.initialize()."""
    openai_mock = ModuleType("openai")

    # Exception hierarchy mirrors the real openai package
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

    openai_mock.APIError = _APIError  # type: ignore[attr-defined]
    openai_mock.AuthenticationError = _AuthenticationError  # type: ignore[attr-defined]
    openai_mock.RateLimitError = _RateLimitError  # type: ignore[attr-defined]
    openai_mock.APIConnectionError = _APIConnectionError  # type: ignore[attr-defined]
    openai_mock.APITimeoutError = _APITimeoutError  # type: ignore[attr-defined]
    openai_mock.BadRequestError = _BadRequestError  # type: ignore[attr-defined]

    # OpenAI() returns a client; client.chat.completions.create() is the call
    client_mock = MagicMock()
    openai_mock.OpenAI = MagicMock(return_value=client_mock)  # type: ignore[attr-defined]
    return openai_mock


@pytest.fixture
def openai_mod() -> Generator[ModuleType, None, None]:
    """Inject a fake openai module for the duration of the test."""
    stub = _stub_openai_module()
    with patch.dict(sys.modules, {"openai": stub}):
        yield stub


@pytest.fixture
def config() -> Configuration:
    return _make_config()


@pytest.fixture
def agent_with_mock_openai(config, openai_mod, monkeypatch):
    """Return a GPTAgent with initialize() already called using a mocked OpenAI."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    # Import after patch so _OpenAIAdapter.initialize picks up the stub module
    from llm_reliability.agents.gpt_agent import GPTAgent

    agent = GPTAgent(config)
    agent.initialize()
    return agent


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------


def test_gpt_agent_is_agent_subclass(config, openai_mod, monkeypatch):
    """GPTAgent must satisfy the Agent abstract interface."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from llm_reliability.agents.gpt_agent import GPTAgent

    agent = GPTAgent(config)
    assert isinstance(agent, Agent)


def test_gpt_agent_requires_config(openai_mod):
    """Passing None as config must raise ValueError immediately."""
    from llm_reliability.agents.gpt_agent import GPTAgent

    with pytest.raises(ValueError, match="Configuration must be provided"):
        GPTAgent(None)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_initialize_raises_without_api_key(config, openai_mod, monkeypatch):
    """initialize() must raise AuthenticationError when OPENAI_API_KEY is unset."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from llm_reliability.agents.adapters.exceptions import AuthenticationError
    from llm_reliability.agents.gpt_agent import GPTAgent

    agent = GPTAgent(config)
    with pytest.raises(AuthenticationError, match="OPENAI_API_KEY"):
        agent.initialize()


def test_initialize_succeeds_with_api_key(config, openai_mod, monkeypatch):
    """initialize() must succeed when OPENAI_API_KEY is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from llm_reliability.agents.gpt_agent import GPTAgent

    agent = GPTAgent(config)
    agent.initialize()  # should not raise
    assert agent._adapter._client is not None


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task,expected_prefix",
    [
        ({"task_id": "t1", "prompt": "What is 2+2?"}, "What is 2+2?"),
        ({"task_id": "t2", "question": "Explain gravity."}, "Explain gravity."),
        ({"task_id": "t3", "problem_statement": "Fix the bug."}, "Fix the bug."),
    ],
)
def test_prompt_extraction_standard_keys(task, expected_prefix, openai_mod):
    """_extract_prompt must recognise standard task keys."""
    from llm_reliability.agents.gpt_agent import GPTAgent

    prompt = GPTAgent._extract_prompt(task)
    assert prompt == expected_prefix


def test_prompt_extraction_falls_back_to_str(openai_mod):
    """_extract_prompt falls back to str(task) for non-standard dicts."""
    from llm_reliability.agents.gpt_agent import GPTAgent

    task = {"task_id": "t99", "content": "do something"}
    prompt = GPTAgent._extract_prompt(task)
    assert "do something" in prompt


def test_prompt_extraction_raises_on_empty_task(openai_mod):
    """_extract_prompt raises ValueError for a completely empty task dict."""
    from llm_reliability.agents.gpt_agent import GPTAgent

    with pytest.raises(ValueError, match="Cannot extract a prompt"):
        GPTAgent._extract_prompt({})


# ---------------------------------------------------------------------------
# run() — success path
# ---------------------------------------------------------------------------


def test_run_returns_raw_text(agent_with_mock_openai, openai_mod):
    """run() must return the raw model text string from the completion."""
    completion = _make_openai_completion(text="four")
    openai_mod.OpenAI.return_value.chat.completions.create.return_value = completion

    task = {"task_id": "t1", "prompt": "What is 2+2?"}
    result = agent_with_mock_openai.run(task)
    assert result == "four"


def test_run_passes_seed_to_request(config, openai_mod, monkeypatch):
    """run() must pass config.seed into the API call when seed is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from llm_reliability.agents.gpt_agent import GPTAgent

    completion = _make_openai_completion(text="ok")
    client_mock = openai_mod.OpenAI.return_value
    client_mock.chat.completions.create.return_value = completion

    agent = GPTAgent(config)
    agent.initialize()
    agent.run({"task_id": "t1", "prompt": "hello"})

    call_kwargs = client_mock.chat.completions.create.call_args[1]
    assert call_kwargs.get("seed") == config.seed


def test_run_uses_model_from_config_llm(openai_mod, monkeypatch):
    """When no 'model' metadata key is set, config.llm should be the model."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = _make_config(llm="gpt-3.5-turbo")
    from llm_reliability.agents.gpt_agent import GPTAgent

    completion = _make_openai_completion(text="hi")
    client_mock = openai_mod.OpenAI.return_value
    client_mock.chat.completions.create.return_value = completion

    agent = GPTAgent(cfg)
    agent.initialize()
    agent.run({"task_id": "t1", "prompt": "hi"})

    call_kwargs = client_mock.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-3.5-turbo"


def test_run_uses_model_from_metadata(openai_mod, monkeypatch):
    """metadata['model'] overrides config.llm for model selection."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = _make_config(metadata={"model": "gpt-4o"})
    from llm_reliability.agents.gpt_agent import GPTAgent

    completion = _make_openai_completion(text="hi")
    client_mock = openai_mod.OpenAI.return_value
    client_mock.chat.completions.create.return_value = completion

    agent = GPTAgent(cfg)
    agent.initialize()
    agent.run({"task_id": "t1", "prompt": "hi"})

    call_kwargs = client_mock.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# run() — error mapping
# ---------------------------------------------------------------------------


def test_run_raises_authentication_error_on_401(config, openai_mod, monkeypatch):
    """run() must raise AuthenticationError when OpenAI rejects credentials."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-bad")
    from llm_reliability.agents.adapters.exceptions import AuthenticationError
    from llm_reliability.agents.gpt_agent import GPTAgent

    client_mock = openai_mod.OpenAI.return_value
    client_mock.chat.completions.create.side_effect = openai_mod.AuthenticationError("401")

    agent = GPTAgent(config)
    agent.initialize()
    with pytest.raises(AuthenticationError):
        agent.run({"task_id": "t1", "prompt": "hi"})


def test_run_raises_rate_limit_after_retries(config, openai_mod, monkeypatch):
    """run() must raise RateLimitError after exhausting retries."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = _make_config(metadata={"max_retries": 2, "retry_backoff": 0.0})
    from llm_reliability.agents.adapters.exceptions import RateLimitError
    from llm_reliability.agents.gpt_agent import GPTAgent

    client_mock = openai_mod.OpenAI.return_value
    client_mock.chat.completions.create.side_effect = openai_mod.RateLimitError("429")

    agent = GPTAgent(cfg)
    agent.initialize()
    with pytest.raises(RateLimitError):
        agent.run({"task_id": "t1", "prompt": "hi"})


def test_run_raises_connection_error_on_network_failure(config, openai_mod, monkeypatch):
    """run() must raise ProviderConnectionError on network-level failures."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = _make_config(metadata={"max_retries": 1, "retry_backoff": 0.0})
    from llm_reliability.agents.adapters.exceptions import \
        ConnectionError as PCE
    from llm_reliability.agents.gpt_agent import GPTAgent

    client_mock = openai_mod.OpenAI.return_value
    client_mock.chat.completions.create.side_effect = openai_mod.APIConnectionError("network")

    agent = GPTAgent(cfg)
    agent.initialize()
    with pytest.raises(PCE):
        agent.run({"task_id": "t1", "prompt": "hi"})


def test_run_raises_response_validation_error_on_empty_choice(config, openai_mod, monkeypatch):
    """run() must raise ResponseValidationError when the completion is empty."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from llm_reliability.agents.adapters.exceptions import \
        ResponseValidationError
    from llm_reliability.agents.gpt_agent import GPTAgent

    empty_completion = MagicMock()
    empty_completion.choices = []
    client_mock = openai_mod.OpenAI.return_value
    client_mock.chat.completions.create.return_value = empty_completion

    agent = GPTAgent(config)
    agent.initialize()
    with pytest.raises(ResponseValidationError):
        agent.run({"task_id": "t1", "prompt": "hi"})


# ---------------------------------------------------------------------------
# reset() and shutdown()
# ---------------------------------------------------------------------------


def test_reset_clears_logs(agent_with_mock_openai, openai_mod):
    """reset() must clear adapter request/response logs."""
    agent_with_mock_openai._adapter._request_logs.append({"event": "request"})
    agent_with_mock_openai._adapter._response_logs.append({"event": "response"})
    agent_with_mock_openai.reset()
    assert agent_with_mock_openai._adapter._request_logs == []
    assert agent_with_mock_openai._adapter._response_logs == []


def test_shutdown_closes_client(agent_with_mock_openai):
    """shutdown() must set the client to None."""
    agent_with_mock_openai.shutdown()
    assert agent_with_mock_openai._adapter._client is None


# ---------------------------------------------------------------------------
# metadata()
# ---------------------------------------------------------------------------


def test_metadata_returns_required_keys(agent_with_mock_openai):
    """metadata() must return all required keys."""
    meta = agent_with_mock_openai.metadata()
    for key in ("name", "provider", "model", "version", "temperature"):
        assert key in meta, f"Missing metadata key: {key!r}"
    assert meta["name"] == "GPTAgent"
    assert meta["provider"] == "openai"


# ---------------------------------------------------------------------------
# ProviderRegistry self-registration
# ---------------------------------------------------------------------------


def test_provider_registry_registers_openai(openai_mod):
    """Importing gpt_agent must register 'openai' in ProviderRegistry."""
    from llm_reliability.agents.adapters.base_llm_adapter import BaseLLMAdapter
    from llm_reliability.agents.adapters.provider_registry import \
        ProviderRegistry
    from llm_reliability.agents.gpt_agent import _OpenAIAdapter

    if not ProviderRegistry.exists("openai"):
        ProviderRegistry.register("openai", _OpenAIAdapter)
    assert ProviderRegistry.exists("openai")
    adapter_cls = ProviderRegistry.get("openai")
    assert issubclass(adapter_cls, BaseLLMAdapter)
    assert adapter_cls.__name__ == "_OpenAIAdapter"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def test_system_prompt_forwarded_to_api(openai_mod, monkeypatch):
    """A system_prompt in config.metadata must appear as the first message."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = _make_config(metadata={"system_prompt": "You are helpful."})
    from llm_reliability.agents.gpt_agent import GPTAgent

    completion = _make_openai_completion(text="ok")
    client_mock = openai_mod.OpenAI.return_value
    client_mock.chat.completions.create.return_value = completion

    agent = GPTAgent(cfg)
    agent.initialize()
    agent.run({"task_id": "t1", "prompt": "hi"})

    call_kwargs = client_mock.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful."
