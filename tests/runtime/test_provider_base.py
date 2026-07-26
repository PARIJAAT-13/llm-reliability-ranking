"""Tests for BaseProvider — Phase 2 provider infrastructure."""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_reliability.agents.adapters.exceptions import (
    AuthenticationError,
)
from llm_reliability.agents.adapters.exceptions import (
    ConnectionError as ProviderConnectionError,
)
from llm_reliability.agents.adapters.exceptions import (
    ContentFilterError,
    ContextLengthExceededError,
    InvalidRequestError,
    NetworkError,
    ProviderError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
    ResponseValidationError,
    TimeoutError,
)
from llm_reliability.agents.adapters.provider_registry import ProviderRegistry
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse
from llm_reliability.configs.config import Configuration
from llm_reliability.runtime.batching import BatchProcessor
from llm_reliability.runtime.cost_accounting import CostTracker
from llm_reliability.runtime.metadata import RuntimeCapabilities, RuntimeMetadata
from llm_reliability.runtime.provider_base import BaseProvider
from llm_reliability.runtime.streaming import TokenStream

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _MinimalConfig:
    """Minimal config stub for testing."""

    def __init__(self, **overrides: Any) -> None:
        self.experiment_name = "test"
        self.benchmark = "test"
        self.agent = "test"
        self.llm = "test-model"
        self.prompt_version = "v1"
        self.dataset_version = "1.0"
        self.seed = 42
        self.repetitions = 1
        self.metadata: dict[str, Any] = {}
        self.perturbations: tuple[str, ...] = ()
        self.fault_injection = False
        self.reliability_weights: Any = None
        self.visualization: Any = None
        self.statistical: Any = None
        for k, v in overrides.items():
            setattr(self, k, v)


def _make_config(**overrides: Any) -> Any:
    return _MinimalConfig(**overrides)


class _RecordingAdapter:
    """Adapter stub that records calls and returns predictable responses."""

    def __init__(self) -> None:
        self._client = object()
        self._request_logs: list[dict[str, Any]] = []
        self._response_logs: list[dict[str, Any]] = []
        self.generate_calls: list[LLMRequest] = []
        self.retry_calls: list[tuple[LLMRequest, int, float]] = []
        self.initialize_called = False
        self.shutdown_called = False
        self._model = "test-model"
        self._temperature = 0.0
        self._max_tokens = 1024

    def initialize(self) -> None:
        self.initialize_called = True

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls.append(request)
        return LLMResponse(
            text=f"Echo: {request.prompt}",
            finish_reason="stop",
            latency_ms=10.0,
            tokens_input=10,
            tokens_output=20,
            model_name="test-model",
            provider="test",
        )

    def retry(
        self, request: LLMRequest, max_attempts: int = 3, backoff_seconds: float = 1.0
    ) -> LLMResponse:
        self.retry_calls.append((request, max_attempts, backoff_seconds))
        return self.generate(request)

    def shutdown(self) -> None:
        self.shutdown_called = True

    def health_check(self) -> bool:
        return True


class _FailingAdapter:
    """Adapter that always raises a specific error."""

    def __init__(self, error: Exception) -> None:
        self._client = object()
        self._request_logs: list[dict[str, Any]] = []
        self._response_logs: list[dict[str, Any]] = []
        self._model = "test-model"
        self._temperature = 0.0
        self._max_tokens = 1024

    def initialize(self) -> None:
        pass

    def generate(self, request: LLMRequest) -> LLMResponse:
        raise self._error

    def retry(
        self, request: LLMRequest, max_attempts: int = 3, backoff_seconds: float = 1.0
    ) -> LLMResponse:
        raise self._error

    def shutdown(self) -> None:
        pass

    def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Concrete test provider
# ---------------------------------------------------------------------------


class _TestProvider(BaseProvider):
    """Minimal concrete provider for testing BaseProvider (adapter pattern)."""

    provider_name: str = "test"
    default_model: str = "test-model"
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    default_requests_per_second: float = 100.0

    def __init__(self, config: Any, adapter: Any | None = None) -> None:
        super().__init__(config)
        self._adapter = adapter or _RecordingAdapter()

    def initialize(self) -> None:
        self._adapter.initialize()
        self._client = getattr(self._adapter, "_client", None)

    def run(self, task: dict[str, Any]) -> Any:
        prompt = self._extract_prompt(task)
        request = self._build_request(prompt)
        self._rate_limiter.acquire()
        response = self._adapter.retry(request, max_attempts=self._max_retries)
        self._track_cost(response)
        return response.text

    def shutdown(self) -> None:
        self._adapter.shutdown()

    def metadata(self) -> dict[str, Any]:
        base = super().metadata()
        base.update({"name": "_TestProvider", "provider": "test"})
        return base

    def _health_check_impl(self) -> bool:
        return self._adapter.health_check()


class _DirectProvider(BaseProvider):
    """Provider that implements _call_api / _parse_response for testing infrastructure."""

    provider_name: str = "direct"
    default_model: str = "d"
    default_requests_per_second: float = 100.0

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._client = object()
        self._call_count = 0

    def initialize(self) -> None:
        self._client = object()

    def _create_client(self) -> Any:
        return object()

    def _build_request_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        return {"model": self._model, "prompt": request.prompt, "echo": True}

    def _call_api(self, kwargs: dict[str, Any]) -> Any:
        self._call_count += 1
        return {"echo": kwargs.get("prompt", ""), "tokens": (10, 20)}

    def _parse_response(self, response: Any, latency_ms: float) -> LLMResponse:
        return LLMResponse(
            text=response["echo"],
            finish_reason="stop",
            latency_ms=latency_ms,
            tokens_input=response["tokens"][0],
            tokens_output=response["tokens"][1],
            model_name=self._model,
            provider="direct",
        )

    def _extract_stream_tokens(self, response: Any) -> Any:
        yield from response.get("tokens", ["a", "b", "c"])

    def run(self, task: dict[str, Any]) -> Any:
        prompt = self._extract_prompt(task)
        request = self._build_request(prompt)
        response = self._call_with_retry(request)
        self._track_cost(response)
        return response.text

    def shutdown(self) -> None:
        self._client = None


# ===================================================================
# Error hierarchy tests
# ===================================================================


class TestErrorHierarchy:
    """Test the extended error hierarchy — all error types are importable,
    have correct is_transient flags, and inherit from ProviderError."""

    def test_provider_error_is_base(self) -> None:
        assert issubclass(AuthenticationError, ProviderError)
        assert issubclass(RateLimitError, ProviderError)
        assert issubclass(TimeoutError, ProviderError)
        assert issubclass(ProviderUnavailableError, ProviderError)
        assert issubclass(QuotaExceededError, ProviderError)
        assert issubclass(InvalidRequestError, ProviderError)
        assert issubclass(NetworkError, ProviderError)
        assert issubclass(ContentFilterError, ProviderError)
        assert issubclass(ContextLengthExceededError, InvalidRequestError)

    def test_transient_flags(self) -> None:
        assert RateLimitError().is_transient is True
        assert TimeoutError().is_transient is True
        assert ProviderUnavailableError().is_transient is True
        assert NetworkError().is_transient is True
        assert AuthenticationError().is_transient is False
        assert QuotaExceededError().is_transient is False
        assert InvalidRequestError().is_transient is False
        assert ContentFilterError().is_transient is False
        assert ContextLengthExceededError().is_transient is False

    def test_new_errors_are_importable_from_adapters(self) -> None:
        from llm_reliability.agents.adapters.exceptions import (
            ContentFilterError,
            ContextLengthExceededError,
            InvalidRequestError,
            NetworkError,
            ProviderUnavailableError,
            QuotaExceededError,
            TimeoutError,
        )

        assert TimeoutError
        assert QuotaExceededError
        assert ProviderUnavailableError
        assert InvalidRequestError
        assert NetworkError
        assert ContentFilterError
        assert ContextLengthExceededError


# ===================================================================
# BaseProvider initialization tests
# ===================================================================


class TestBaseProviderInit:
    """Test BaseProvider configuration parsing and setup."""

    def test_raises_without_config(self) -> None:
        with pytest.raises(ValueError, match="Configuration must be provided"):
            _TestProvider(None)  # type: ignore[arg-type]

    def test_accepts_minimal_config(self) -> None:
        provider = _TestProvider(_make_config())
        assert provider._model == "test-model"
        assert provider._temperature == 0.0
        assert provider._max_tokens == 1024
        assert provider._top_p == 1.0
        assert provider._rate_limiter is not None
        assert provider._cost_tracker is not None
        assert provider._version == "1.0"

    def test_parses_metadata_overrides(self) -> None:
        config = _make_config(
            metadata={
                "model": "custom-model",
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 0.9,
                "system_prompt": "You are a helpful assistant.",
                "max_retries": 5,
                "retry_backoff": 2.0,
            }
        )
        provider = _TestProvider(config)
        assert provider._model == "custom-model"
        assert provider._temperature == 0.7
        assert provider._max_tokens == 2048
        assert provider._top_p == 0.9
        assert provider._system_prompt == "You are a helpful assistant."
        assert provider._max_retries == 5
        assert provider._retry_backoff == 2.0

    def test_defaults_when_metadata_empty(self) -> None:
        provider = _TestProvider(_make_config(metadata={}))
        assert provider._model == "test-model"
        assert provider._temperature == 0.0
        assert provider._max_tokens == 1024
        assert provider._top_p == 1.0
        assert provider._max_retries == 3
        assert provider._retry_backoff == 1.0

    def test_falls_back_to_class_defaults(self) -> None:
        config = _make_config(llm="")
        provider = _TestProvider(config)
        assert provider._model == "test-model"

    def test_rate_limiter_configured(self) -> None:
        config = _make_config(metadata={"requests_per_second": 50.0})
        provider = _TestProvider(config)
        assert provider._requests_per_second == 50.0

    def test_client_is_none_after_init(self) -> None:
        provider = _TestProvider(_make_config())
        assert provider._client is None

    def test_provider_name_class_attr(self) -> None:
        assert _TestProvider.provider_name == "test"


# ===================================================================
# BaseProvider lifecycle tests
# ===================================================================


class TestBaseProviderLifecycle:
    """Test initialize, run, shutdown lifecycle."""

    def test_initialize_calls_adapter(self) -> None:
        adapter = _RecordingAdapter()
        provider = _TestProvider(_make_config(), adapter=adapter)
        provider.initialize()
        assert adapter.initialize_called
        assert provider._client is not None

    def test_run_returns_text(self) -> None:
        adapter = _RecordingAdapter()
        provider = _TestProvider(_make_config(seed=42), adapter=adapter)
        provider.initialize()
        result = provider.run({"prompt": "Hello"})
        assert result == "Echo: Hello"

    def test_run_adds_cost_entry(self) -> None:
        adapter = _RecordingAdapter()
        provider = _TestProvider(_make_config(seed=42), adapter=adapter)
        provider.initialize()
        provider.run({"prompt": "Hello"})
        assert provider._cost_tracker.entry_count == 1

    def test_shutdown_calls_adapter(self) -> None:
        adapter = _RecordingAdapter()
        provider = _TestProvider(_make_config(), adapter=adapter)
        provider.shutdown()
        assert adapter.shutdown_called

    def test_reset_clears_cost_tracker(self) -> None:
        adapter = _RecordingAdapter()
        provider = _TestProvider(_make_config(), adapter=adapter)
        provider.initialize()
        provider.run({"prompt": "Hello"})
        assert provider._cost_tracker.entry_count == 1
        provider.reset()
        assert provider._cost_tracker.entry_count == 0

    def test_reset_calls_super_reset(self) -> None:
        adapter = _RecordingAdapter()
        provider = _TestProvider(_make_config(), adapter=adapter)
        provider.initialize()
        provider.run({"prompt": "Hello"})
        assert provider.cost_summary()["call_count"] == 1
        provider.reset()
        assert provider.cost_summary()["call_count"] == 0


# ===================================================================
# Cost tracking tests
# ===================================================================


class TestCostTracking:
    """Test the cost accounting integration."""

    def test_track_cost_adds_entry(self) -> None:
        provider = _TestProvider(_make_config())
        provider.initialize()
        response = LLMResponse(
            text="test",
            finish_reason="stop",
            latency_ms=10.0,
            tokens_input=100,
            tokens_output=50,
            model_name="test-model",
            provider="test",
        )
        provider._track_cost(response)
        assert provider._cost_tracker.entry_count == 1
        entry = provider._cost_tracker.entries[0]
        assert entry.input_tokens == 100
        assert entry.output_tokens == 50
        assert entry.provider == "test"

    def test_cost_summary_returns_dict(self) -> None:
        provider = _TestProvider(_make_config())
        provider.initialize()
        response1 = LLMResponse(
            text="a",
            finish_reason="stop",
            latency_ms=10.0,
            tokens_input=100,
            tokens_output=50,
            model_name="m",
            provider="test",
        )
        response2 = LLMResponse(
            text="b",
            finish_reason="stop",
            latency_ms=20.0,
            tokens_input=200,
            tokens_output=100,
            model_name="m",
            provider="test",
        )
        provider._track_cost(response1)
        provider._track_cost(response2)
        summary = provider.cost_summary()
        assert summary["call_count"] == 2
        assert summary["total_input_tokens"] == 300
        assert summary["total_output_tokens"] == 150
        assert summary["total_tokens"] == 450

    def test_run_tracks_cost_automatically(self) -> None:
        adapter = _RecordingAdapter()
        provider = _TestProvider(_make_config(), adapter=adapter)
        provider.initialize()
        provider.run({"prompt": "Hello"})
        summary = provider.cost_summary()
        assert summary["call_count"] == 1

    def test_cost_tracker_property(self) -> None:
        provider = _TestProvider(_make_config())
        assert isinstance(provider.cost_tracker, CostTracker)


# ===================================================================
# Prompt extraction tests
# ===================================================================


class TestPromptExtraction:
    """Test the shared _extract_prompt utility."""

    def test_extracts_from_prompt_key(self) -> None:
        result = _TestProvider._extract_prompt({"prompt": "Hello"})
        assert result == "Hello"

    def test_extracts_from_question_key(self) -> None:
        result = _TestProvider._extract_prompt({"question": "What is 2+2?"})
        assert result == "What is 2+2?"

    def test_extracts_from_problem_statement_key(self) -> None:
        result = _TestProvider._extract_prompt({"problem_statement": "Solve x+2=5"})
        assert result == "Solve x+2=5"

    def test_priority_order_prompt_first(self) -> None:
        result = _TestProvider._extract_prompt(
            {
                "prompt": "from prompt",
                "question": "from question",
                "problem_statement": "from problem",
            }
        )
        assert result == "from prompt"

    def test_priority_question_second(self) -> None:
        result = _TestProvider._extract_prompt(
            {
                "question": "from question",
                "problem_statement": "from problem",
            }
        )
        assert result == "from question"

    def test_falls_back_to_str_of_task(self) -> None:
        result = _TestProvider._extract_prompt({"custom_key": "value"})
        assert "{'custom_key': 'value'}" in result or '"custom_key": "value"' in result

    def test_raises_on_empty_dict(self) -> None:
        with pytest.raises(ValueError, match="Cannot extract a prompt"):
            _TestProvider._extract_prompt({})

    def test_strips_whitespace(self) -> None:
        result = _TestProvider._extract_prompt({"prompt": "  Hello  "})
        assert result == "Hello"

    def test_skips_blank_values(self) -> None:
        result = _TestProvider._extract_prompt({"prompt": "", "question": "real question"})
        assert result == "real question"


# ===================================================================
# Metadata tests
# ===================================================================


class TestMetadata:
    """Test metadata and runtime_metadata reporting."""

    def test_metadata_returns_dict(self) -> None:
        provider = _TestProvider(_make_config(seed=42))
        meta = provider.metadata()
        assert isinstance(meta, dict)
        assert meta["name"] == "_TestProvider"
        assert meta["provider"] == "test"
        assert meta["model"] == "test-model"
        assert meta["version"] == "1.0"
        assert meta["seed"] == 42

    def test_runtime_metadata_returns_runtimemetadata(self) -> None:
        provider = _TestProvider(_make_config())
        rm = provider.runtime_metadata()
        assert isinstance(rm, RuntimeMetadata)
        assert rm.runtime_name == "_TestProvider"
        assert rm.backend == "test"
        assert rm.capabilities is not None

    def test_runtime_metadata_includes_inference_params(self) -> None:
        provider = _TestProvider(_make_config())
        rm = provider.runtime_metadata()
        assert "temperature" in rm.inference_parameters
        assert "max_tokens" in rm.inference_parameters

    def test_detect_capabilities(self) -> None:
        provider = _TestProvider(_make_config())
        caps = provider._detect_capabilities()
        assert isinstance(caps, RuntimeCapabilities)
        assert caps.batch_inference is True
        assert caps.streaming is True

    def test_detect_gpu(self) -> None:
        provider = _TestProvider(_make_config())
        result = provider._detect_gpu()
        assert isinstance(result, bool)


# ===================================================================
# Health check tests
# ===================================================================


class TestHealthCheck:
    """Test health check behavior."""

    def test_health_check_delegates_to_adapter(self) -> None:
        adapter = _RecordingAdapter()
        provider = _TestProvider(_make_config(), adapter=adapter)
        provider.initialize()
        result = provider.health_check()
        assert result is True

    def test_health_check_returns_false_when_no_client(self) -> None:
        provider = _TestProvider(_make_config())
        result = provider.health_check()
        assert result is False

    def test_health_check_false_on_exception(self) -> None:
        adapter = _RecordingAdapter()
        original = adapter.health_check
        adapter.health_check = lambda: (_ for _ in ()).throw(Exception("fail"))  # type: ignore[method-assign]
        provider = _TestProvider(_make_config(), adapter=adapter)
        provider.initialize()
        result = provider.health_check()
        assert result is False
        adapter.health_check = original


# ===================================================================
# Streaming tests
# ===================================================================


class TestStreaming:
    """Test the BaseProvider.stream() method."""

    def test_stream_returns_tokenstream(self) -> None:
        provider = _DirectProvider(_make_config())
        provider.initialize()
        request = LLMRequest(prompt="Hello")
        stream = provider.stream(request)
        assert isinstance(stream, TokenStream)

    def test_stream_yields_tokens(self) -> None:
        provider = _DirectProvider(_make_config())
        provider.initialize()
        request = LLMRequest(prompt="Hello")
        stream = provider.stream(request)
        tokens = list(stream)
        assert len(tokens) >= 0

    def test_stream_timeout(self) -> None:
        provider = _DirectProvider(_make_config())
        provider.initialize()
        request = LLMRequest(prompt="Hello")
        stream = provider.stream(request, timeout=0.001)
        collected = list(stream)
        assert isinstance(collected, list)


# ===================================================================
# Batching tests
# ===================================================================


class TestBatching:
    """Test the BaseProvider.batch() method."""

    def test_batch_returns_list_of_responses(self) -> None:
        provider = _DirectProvider(_make_config())
        provider.initialize()
        tasks = [
            {"prompt": "Task 1"},
            {"prompt": "Task 2"},
            {"prompt": "Task 3"},
        ]
        results = provider.batch(tasks, max_batch_size=2)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, LLMResponse)

    def test_batch_handles_empty_list(self) -> None:
        provider = _DirectProvider(_make_config())
        provider.initialize()
        results = provider.batch([], max_batch_size=2)
        assert results == []

    def test_batch_respects_max_batch_size(self) -> None:
        provider = _DirectProvider(_make_config())
        provider.initialize()
        tasks = [{"prompt": f"Task {i}"} for i in range(10)]
        results = provider.batch(tasks, max_batch_size=3)
        assert len(results) == 10

    def test_batch_produces_responses_with_all_fields(self) -> None:
        provider = _DirectProvider(_make_config())
        provider.initialize()
        tasks = [{"prompt": f"Task {i}"} for i in range(3)]
        results = provider.batch(tasks, max_batch_size=2)
        for r in results:
            assert r.text in {"Task 0", "Task 1", "Task 2"}
            assert r.finish_reason == "stop"
            assert isinstance(r.latency_ms, float)
            assert r.tokens_input == 10
            assert r.tokens_output == 20
            assert r.model_name == provider._model
            assert r.provider == "direct"


# ===================================================================
# Retry tests
# ===================================================================


class TestRetry:
    """Test the BaseProvider.retry() method uses the call chain correctly."""

    def test_retry_succeeds_on_first_attempt(self) -> None:
        provider = _DirectProvider(_make_config())
        provider.initialize()
        request = LLMRequest(prompt="Hello")
        response = provider.retry(request, max_attempts=3, backoff_seconds=0.01)
        assert response.text == "Hello"

    def test_retry_eventually_succeeds(self) -> None:
        class _FailingThenSucceeding(_DirectProvider):
            def __init__(self, config: Any) -> None:
                super().__init__(config)
                self._call_count = 0

            def _call_api(self, kwargs: dict[str, Any]) -> Any:
                self._call_count += 1
                if self._call_count < 3:
                    raise RateLimitError("rate limited")
                return {"echo": kwargs.get("prompt", ""), "tokens": (10, 20)}

        provider = _FailingThenSucceeding(_make_config())
        provider.initialize()
        request = LLMRequest(prompt="Hello")
        response = provider.retry(request, max_attempts=5, backoff_seconds=0.01)
        assert response.text == "Hello"
        assert provider._call_count == 3

    def test_retry_raises_on_non_transient(self) -> None:
        class _FailingAuth(_DirectProvider):
            def _call_api(self, kwargs: dict[str, Any]) -> Any:
                raise AuthenticationError("bad key")

        provider = _FailingAuth(_make_config())
        provider.initialize()
        request = LLMRequest(prompt="Hello")
        with pytest.raises(AuthenticationError):
            provider.retry(request, max_attempts=3, backoff_seconds=0.01)


# ===================================================================
# ProviderRegistry compatibility tests
# ===================================================================


class TestProviderRegistryCompatibility:
    """Test that adapters are still properly registered in ProviderRegistry."""

    _REGS: list[tuple[str, str, str]] = [
        ("openai", "gpt_agent", "_OpenAIAdapter"),
        ("anthropic", "anthropic_agent", "_AnthropicAdapter"),
        ("google", "gemini_agent", "_GeminiAdapter"),
        ("deepseek", "deepseek_agent", "_DeepSeekAdapter"),
        ("mistral", "mistral_agent", "_MistralAdapter"),
        ("cohere", "cohere_agent", "_CohereAdapter"),
        ("ollama", "ollama_agent", "_OllamaAdapter"),
    ]

    @staticmethod
    def _ensure_registration(key: str, module_name: str, cls_name: str) -> type:
        import importlib

        mod = importlib.import_module(f"llm_reliability.agents.{module_name}")
        cls = getattr(mod, cls_name)
        if not ProviderRegistry.exists(key):
            ProviderRegistry.register(key, cls)
        return cls

    def test_openai_adapter_registered(self) -> None:
        cls = self._ensure_registration("openai", "gpt_agent", "_OpenAIAdapter")
        assert ProviderRegistry.get("openai") is cls

    def test_anthropic_adapter_registered(self) -> None:
        cls = self._ensure_registration("anthropic", "anthropic_agent", "_AnthropicAdapter")
        assert ProviderRegistry.get("anthropic") is cls

    def test_gemini_adapter_registered(self) -> None:
        cls = self._ensure_registration("google", "gemini_agent", "_GeminiAdapter")
        assert ProviderRegistry.get("google") is cls

    def test_deepseek_adapter_registered(self) -> None:
        self._ensure_registration("deepseek", "deepseek_agent", "_DeepSeekAdapter")
        assert ProviderRegistry.exists("deepseek")

    def test_mistral_adapter_registered(self) -> None:
        self._ensure_registration("mistral", "mistral_agent", "_MistralAdapter")
        assert ProviderRegistry.exists("mistral")

    def test_cohere_adapter_registered(self) -> None:
        self._ensure_registration("cohere", "cohere_agent", "_CohereAdapter")
        assert ProviderRegistry.exists("cohere")

    def test_ollama_adapter_registered(self) -> None:
        self._ensure_registration("ollama", "ollama_agent", "_OllamaAdapter")
        assert ProviderRegistry.exists("ollama")


# ===================================================================
# Runtime registry compatibility tests
# ===================================================================


class TestRuntimeRegistryCompatibility:
    """Test that all agents are properly registered in RuntimeRegistry."""

    _REGS: list[tuple[str, str, str]] = [
        ("gpt", "gpt_agent", "GPTAgent"),
        ("anthropic", "anthropic_agent", "AnthropicAgent"),
        ("gemini", "gemini_agent", "GeminiAgent"),
        ("deepseek", "deepseek_agent", "DeepSeekAgent"),
        ("mistral", "mistral_agent", "MistralAgent"),
        ("cohere", "cohere_agent", "CohereAgent"),
        ("ollama", "ollama_agent", "OllamaAgent"),
        ("mock", "mock_agent", "MockAgent"),
    ]

    @staticmethod
    def _ensure_registration(key: str, module_name: str, cls_name: str) -> type:
        import importlib

        from llm_reliability.runtime.registry import RuntimeRegistry

        mod = importlib.import_module(f"llm_reliability.agents.{module_name}")
        cls = getattr(mod, cls_name)
        if not RuntimeRegistry.exists(key):
            RuntimeRegistry.register(key, cls)
        return cls

    def test_gpt_registered(self) -> None:
        self._ensure_registration("gpt", "gpt_agent", "GPTAgent")
        from llm_reliability.runtime.registry import RuntimeRegistry

        assert RuntimeRegistry.exists("gpt")

    def test_anthropic_registered(self) -> None:
        self._ensure_registration("anthropic", "anthropic_agent", "AnthropicAgent")
        from llm_reliability.runtime.registry import RuntimeRegistry

        assert RuntimeRegistry.exists("anthropic")

    def test_gemini_registered(self) -> None:
        self._ensure_registration("gemini", "gemini_agent", "GeminiAgent")
        from llm_reliability.runtime.registry import RuntimeRegistry

        assert RuntimeRegistry.exists("gemini")

    def test_deepseek_registered(self) -> None:
        self._ensure_registration("deepseek", "deepseek_agent", "DeepSeekAgent")
        from llm_reliability.runtime.registry import RuntimeRegistry

        assert RuntimeRegistry.exists("deepseek")

    def test_mistral_registered(self) -> None:
        self._ensure_registration("mistral", "mistral_agent", "MistralAgent")
        from llm_reliability.runtime.registry import RuntimeRegistry

        assert RuntimeRegistry.exists("mistral")

    def test_cohere_registered(self) -> None:
        self._ensure_registration("cohere", "cohere_agent", "CohereAgent")
        from llm_reliability.runtime.registry import RuntimeRegistry

        assert RuntimeRegistry.exists("cohere")

    def test_ollama_registered(self) -> None:
        self._ensure_registration("ollama", "ollama_agent", "OllamaAgent")
        from llm_reliability.runtime.registry import RuntimeRegistry

        assert RuntimeRegistry.exists("ollama")

    def test_mock_registered(self) -> None:
        self._ensure_registration("mock", "mock_agent", "MockAgent")
        from llm_reliability.runtime.registry import RuntimeRegistry

        assert RuntimeRegistry.exists("mock")


# ===================================================================
# Stream / batch from Runtime interface tests
# ===================================================================


class TestRuntimeInterface:
    """Test that BaseProvider properly integrates with the Runtime interface."""

    def test_is_runtime_subclass(self) -> None:
        from llm_reliability.runtime.interface import Runtime

        assert issubclass(BaseProvider, Runtime)

    def test_execute_delegates_to_run(self) -> None:
        adapter = _RecordingAdapter()
        provider = _TestProvider(_make_config(), adapter=adapter)
        provider.initialize()
        result = provider.execute({"prompt": "Hello"})
        assert result == "Echo: Hello"

    def test_measure_latency_returns_tuple(self) -> None:
        adapter = _RecordingAdapter()
        provider = _TestProvider(_make_config(), adapter=adapter)
        provider.initialize()
        output, latency = provider.measure_latency({"prompt": "Hello"})
        assert output == "Echo: Hello"
        assert latency > 0

    def test_health_check_default_true(self) -> None:
        adapter = _RecordingAdapter()
        provider = _TestProvider(_make_config(), adapter=adapter)
        provider.initialize()
        assert provider.health_check() is True


# ===================================================================
# Error mapping tests
# ===================================================================


class TestErrorMapping:
    """Test the BaseProvider._map_provider_error method."""

    def test_passes_through_provider_error(self) -> None:
        provider = _TestProvider(_make_config())
        err = RateLimitError("test")
        result = provider._map_provider_error(err)
        assert result is err

    def test_wraps_unknown_exception(self) -> None:
        provider = _TestProvider(_make_config())
        result = provider._map_provider_error(ValueError("test"))
        assert isinstance(result, ProviderError)
        assert "test" in str(result)


# ===================================================================
# Request builder tests
# ===================================================================


class TestRequestBuilder:
    """Test the BaseProvider._build_request and _build_request_kwargs methods."""

    def test_build_request_creates_llmrequest(self) -> None:
        provider = _TestProvider(_make_config(seed=42))
        request = provider._build_request("Hello")
        assert isinstance(request, LLMRequest)
        assert request.prompt == "Hello"
        assert request.temperature == 0.0
        assert request.max_tokens == 1024

    def test_build_request_includes_system_prompt(self) -> None:
        config = _make_config(metadata={"system_prompt": "Be helpful."}, seed=42)
        provider = _TestProvider(config)
        request = provider._build_request("Hello")
        assert request.system_prompt == "Be helpful."

    def test_build_request_kwargs_basic(self) -> None:
        provider = _TestProvider(_make_config())
        request = LLMRequest(prompt="Hello", temperature=0.5, max_tokens=100)
        kwargs = provider._build_request_kwargs(request)
        assert kwargs["model"] == "test-model"
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 100
        assert "messages" in kwargs

    def test_build_request_kwargs_includes_system(self) -> None:
        config = _make_config(metadata={"system_prompt": "Be concise."})
        provider = _TestProvider(config)
        request = LLMRequest(prompt="Hello", system_prompt="Hi")
        kwargs = provider._build_request_kwargs(request)
        messages = kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Hi"

    def test_build_request_kwargs_includes_stop(self) -> None:
        provider = _TestProvider(_make_config())
        request = LLMRequest(prompt="Hello", stop_sequences=["\n", "END"])
        kwargs = provider._build_request_kwargs(request)
        assert kwargs["stop"] == ["\n", "END"]


# ===================================================================
# Agent class hierarchy tests
# ===================================================================


class TestAgentClassHierarchy:
    """Test that all refactored agent classes inherit from BaseProvider."""

    def test_gpt_agent_is_baseprovider(self) -> None:
        from llm_reliability.agents.gpt_agent import GPTAgent

        assert issubclass(GPTAgent, BaseProvider)

    def test_anthropic_agent_is_baseprovider(self) -> None:
        from llm_reliability.agents.anthropic_agent import AnthropicAgent

        assert issubclass(AnthropicAgent, BaseProvider)

    def test_gemini_agent_is_baseprovider(self) -> None:
        from llm_reliability.agents.gemini_agent import GeminiAgent

        assert issubclass(GeminiAgent, BaseProvider)

    def test_cohere_agent_is_baseprovider(self) -> None:
        from llm_reliability.agents.cohere_agent import CohereAgent

        assert issubclass(CohereAgent, BaseProvider)

    def test_mistral_agent_is_baseprovider(self) -> None:
        from llm_reliability.agents.mistral_agent import MistralAgent

        assert issubclass(MistralAgent, BaseProvider)

    def test_deepseek_agent_is_baseprovider(self) -> None:
        from llm_reliability.agents.deepseek_agent import DeepSeekAgent

        assert issubclass(DeepSeekAgent, BaseProvider)

    def test_ollama_agent_is_baseprovider(self) -> None:
        from llm_reliability.agents.ollama_agent import OllamaAgent

        assert issubclass(OllamaAgent, BaseProvider)

    def test_mock_agent_is_baseprovider(self) -> None:
        from llm_reliability.agents.mock_agent import MockAgent

        assert issubclass(MockAgent, BaseProvider)

    def test_all_refactored_agents(self) -> None:
        from llm_reliability.agents import (
            anthropic_agent,
            azure_openai_agent,
            bedrock_agent,
            cerebras_agent,
            cohere_agent,
            deepseek_agent,
            fireworks_agent,
            gemini_agent,
            gpt_agent,
            groq_agent,
            hf_agent,
            litellm_agent,
            llama_agent,
            llama_cpp_agent,
            mistral_agent,
            mock_agent,
            nim_agent,
            ollama_agent,
            openrouter_agent,
            perplexity_agent,
            qwen_agent,
            sambanova_agent,
            sglang_agent,
            together_agent,
            vertex_agent,
            vllm_agent,
            xai_agent,
        )

        agent_classes = [
            gpt_agent.GPTAgent,
            anthropic_agent.AnthropicAgent,
            gemini_agent.GeminiAgent,
            cohere_agent.CohereAgent,
            mistral_agent.MistralAgent,
            deepseek_agent.DeepSeekAgent,
            ollama_agent.OllamaAgent,
            llama_agent.LlamaAgent,
            llama_cpp_agent.LlamaCppAgent,
            vllm_agent.VLLMAgent,
            sglang_agent.SGLangAgent,
            hf_agent.HuggingFaceAgent,
            openrouter_agent.OpenRouterAgent,
            together_agent.TogetherAgent,
            groq_agent.GroqAgent,
            fireworks_agent.FireworksAgent,
            perplexity_agent.PerplexityAgent,
            xai_agent.XAIAgent,
            sambanova_agent.SambaNovaAgent,
            cerebras_agent.CerebrasAgent,
            nim_agent.NIMAgent,
            bedrock_agent.BedrockAgent,
            azure_openai_agent.AzureOpenAIAgent,
            vertex_agent.VertexAgent,
            litellm_agent.LiteLLMAgent,
            qwen_agent.QwenAgent,
            mock_agent.MockAgent,
        ]
        for cls in agent_classes:
            assert issubclass(
                cls, BaseProvider
            ), f"{cls.__name__} does not inherit from BaseProvider"


# ===================================================================
# Runtime __init__ exports
# ===================================================================


class TestRuntimeExports:
    """Test that BaseProvider is properly exported from runtime package."""

    def test_base_provider_exported(self) -> None:
        from llm_reliability.runtime import BaseProvider as RuntimeBaseProvider

        assert RuntimeBaseProvider is BaseProvider

    def test_all_runtime_exports(self) -> None:
        from llm_reliability.runtime import __all__

        assert "BaseProvider" in __all__
        assert "Runtime" in __all__
        assert "RuntimeRegistry" in __all__


# ===================================================================
# Provider agent instantiation tests (smoke)
# ===================================================================


class TestAgentInstantiation:
    """Smoke tests — verify agents can be created with config."""

    def test_smoke_mock_agent(self) -> None:
        from llm_reliability.agents.mock_agent import MockAgent

        config = _make_config(metadata={})
        agent = MockAgent(config)
        assert isinstance(agent, BaseProvider)

    def test_smoke_gpt_agent_config(self) -> None:
        from llm_reliability.agents.gpt_agent import GPTAgent

        config = _make_config(metadata={})
        agent = GPTAgent(config)
        assert isinstance(agent, BaseProvider)

    def test_smoke_anthropic_agent(self) -> None:
        from llm_reliability.agents.anthropic_agent import AnthropicAgent

        config = _make_config(metadata={})
        agent = AnthropicAgent(config)
        assert isinstance(agent, BaseProvider)

    def test_smoke_ollama_agent(self) -> None:
        from llm_reliability.agents.ollama_agent import OllamaAgent

        config = _make_config(metadata={})
        agent = OllamaAgent(config)
        assert isinstance(agent, BaseProvider)

    def test_smoke_mistral_agent(self) -> None:
        from llm_reliability.agents.mistral_agent import MistralAgent

        config = _make_config(metadata={})
        agent = MistralAgent(config)
        assert isinstance(agent, BaseProvider)

    def test_smoke_cohere_agent(self) -> None:
        from llm_reliability.agents.cohere_agent import CohereAgent

        config = _make_config(metadata={})
        agent = CohereAgent(config)
        assert isinstance(agent, BaseProvider)


# ===================================================================
# Edge case tests
# ===================================================================


class TestEdgeCases:
    """Test edge cases for BaseProvider."""

    def test_run_with_empty_task_raises(self) -> None:
        adapter = _RecordingAdapter()
        provider = _TestProvider(_make_config(), adapter=adapter)
        provider.initialize()
        with pytest.raises(ValueError, match="Cannot extract a prompt"):
            provider.run({})

    def test_initialize_raises_when_no_api_key(self) -> None:
        class _KeyedProvider(BaseProvider):
            provider_name = "keyed"
            default_model = "m"
            api_key_env = "REQUIRED_KEY"

            def initialize(self) -> None:
                from llm_reliability.agents.adapters.exceptions import (
                    AuthenticationError,
                )

                if not __import__("os").environ.get(self.api_key_env):
                    raise AuthenticationError(f"{self.api_key_env} not set")

        provider = _KeyedProvider(_make_config())
        with pytest.raises(AuthenticationError):
            provider.initialize()

    def test_stream_with_none_generator(self) -> None:
        ts = TokenStream()
        assert list(ts) == []

    def test_tokenstream_cancel(self) -> None:
        def gen() -> Any:
            yield "a"
            yield "b"
            yield "c"

        ts = TokenStream(generator=gen())
        ts.cancel()
        assert list(ts) == []

    def test_tokenstream_timed_out_property(self) -> None:
        import time as _time

        def _gen() -> Any:
            for i in range(100):
                yield str(i)

        ts = TokenStream(generator=_gen(), timeout=0.001)
        it = iter(ts)
        token = next(it)
        assert token == "0"
        _time.sleep(0.05)
        assert ts.timed_out, f"expected True, got {ts.timed_out}"

    def test_batch_processor_rejects_invalid_size(self) -> None:
        with pytest.raises(ValueError, match="max_batch_size"):
            BatchProcessor(executor=MagicMock(), max_batch_size=0)
