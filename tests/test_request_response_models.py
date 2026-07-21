"""Tests for LLMRequest and LLMResponse models."""

import pytest

from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse


# ── LLMRequest ───────────────────────────────────────────────────────────────

def test_request_valid():
    req = LLMRequest(prompt="Hello")
    assert req.prompt == "Hello"
    assert req.temperature == 0.0
    assert req.max_tokens == 1024
    assert req.top_p == 1.0


def test_request_blank_prompt_rejected():
    with pytest.raises(Exception):
        LLMRequest(prompt="   ")


def test_request_temperature_bounds():
    LLMRequest(prompt="p", temperature=0.0)
    LLMRequest(prompt="p", temperature=2.0)
    with pytest.raises(Exception):
        LLMRequest(prompt="p", temperature=2.1)
    with pytest.raises(Exception):
        LLMRequest(prompt="p", temperature=-0.1)


def test_request_max_tokens_must_be_positive():
    with pytest.raises(Exception):
        LLMRequest(prompt="p", max_tokens=0)


def test_request_serialization_roundtrip():
    req = LLMRequest(prompt="test", seed=42, stop_sequences=["END"])
    assert LLMRequest.model_validate_json(req.model_dump_json()) == req


# ── LLMResponse ──────────────────────────────────────────────────────────────

def test_response_valid():
    resp = LLMResponse(
        text="hello",
        finish_reason="stop",
        latency_ms=120.5,
        tokens_input=10,
        tokens_output=5,
        model_name="gpt-4o",
        provider="openai",
    )
    assert resp.text == "hello"
    assert resp.latency_ms == 120.5


def test_response_latency_must_be_non_negative():
    with pytest.raises(Exception):
        LLMResponse(
            text="x",
            finish_reason="stop",
            latency_ms=-1.0,
            tokens_input=1,
            tokens_output=1,
            model_name="m",
            provider="p",
        )


def test_response_serialization_roundtrip():
    resp = LLMResponse(
        text="world",
        finish_reason="length",
        latency_ms=0.0,
        tokens_input=5,
        tokens_output=2,
        model_name="claude-3",
        provider="anthropic",
        metadata={"key": "val"},
    )
    assert LLMResponse.model_validate_json(resp.model_dump_json()) == resp
