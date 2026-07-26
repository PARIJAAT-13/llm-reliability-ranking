"""Tests for BaseLLMAdapter."""

import pytest

from llm_reliability.agents.adapters.base_llm_adapter import BaseLLMAdapter
from llm_reliability.agents.adapters.exceptions import (
    ProviderError,
    RequestValidationError,
    ResponseValidationError,
)
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse
from llm_reliability.configs.config import Configuration


@pytest.fixture
def config():
    return Configuration(
        experiment_name="test",
        benchmark="mock",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
    )


def _make_response(**overrides) -> LLMResponse:
    defaults = dict(
        text="hello",
        finish_reason="stop",
        latency_ms=50.0,
        tokens_input=5,
        tokens_output=3,
        model_name="test-model",
        provider="test",
    )
    defaults.update(overrides)
    return LLMResponse(**defaults)


class ConcreteAdapter(BaseLLMAdapter):
    """A minimal concrete implementation for testing."""

    def initialize(self) -> None:
        pass

    def generate(self, request: LLMRequest) -> LLMResponse:
        return _make_response()

    def shutdown(self) -> None:
        pass

    def provider_metadata(self) -> dict:
        return {"provider": "test"}

    def health_check(self) -> bool:
        return True


class FailingAdapter(BaseLLMAdapter):
    """Always raises ProviderError from generate()."""

    def initialize(self) -> None:
        pass

    def generate(self, request: LLMRequest) -> LLMResponse:
        raise ProviderError("service unavailable")

    def shutdown(self) -> None:
        pass

    def provider_metadata(self) -> dict:
        return {}

    def health_check(self) -> bool:
        return False


class BlankResponseAdapter(BaseLLMAdapter):
    """Returns a response with blank text — violates the response contract."""

    def initialize(self) -> None:
        pass

    def generate(self, request: LLMRequest) -> LLMResponse:
        return _make_response(text="   ")

    def shutdown(self) -> None:
        pass

    def provider_metadata(self) -> dict:
        return {}

    def health_check(self) -> bool:
        return True


# ── Tests ────────────────────────────────────────────────────────────────────


def test_adapter_implements_interface(config):
    adapter = ConcreteAdapter(config)
    assert isinstance(adapter, BaseLLMAdapter)


def test_adapter_requires_configuration():
    with pytest.raises(ValueError):
        ConcreteAdapter(None)


def test_validate_request_rejects_wrong_type(config):
    adapter = ConcreteAdapter(config)
    with pytest.raises(RequestValidationError):
        adapter.validate_request("not a request")


def test_validate_response_rejects_blank_text(config):
    adapter = ConcreteAdapter(config)
    with pytest.raises(ResponseValidationError):
        adapter.validate_response(_make_response(text="  "))


def test_measure_latency_success(config):
    adapter = ConcreteAdapter(config)
    req = LLMRequest(prompt="hello")
    resp, latency = adapter.measure_latency(req)
    assert resp.text == "hello"
    assert latency >= 0.0


def test_logging_captures_requests_and_responses(config):
    adapter = ConcreteAdapter(config)
    req = LLMRequest(prompt="test prompt")
    adapter.measure_latency(req)

    assert len(adapter._request_logs) == 1
    assert len(adapter._response_logs) == 1
    assert adapter._request_logs[0]["event"] == "request"
    assert adapter._response_logs[0]["event"] == "response"


def test_retry_succeeds_on_first_attempt(config):
    adapter = ConcreteAdapter(config)
    req = LLMRequest(prompt="hello")
    resp = adapter.retry(req, max_attempts=3)
    assert resp.text == "hello"


def test_retry_exhausts_and_raises(config):
    adapter = FailingAdapter(config)
    req = LLMRequest(prompt="hello")
    with pytest.raises(ProviderError, match="service unavailable"):
        adapter.retry(req, max_attempts=2, backoff_seconds=0.0)


def test_health_check(config):
    adapter = ConcreteAdapter(config)
    assert adapter.health_check() is True

    failing = FailingAdapter(config)
    assert failing.health_check() is False


def test_provider_metadata(config):
    adapter = ConcreteAdapter(config)
    meta = adapter.provider_metadata()
    assert isinstance(meta, dict)


def test_abstract_class_cannot_be_instantiated(config):
    with pytest.raises(TypeError):
        BaseLLMAdapter(config)
