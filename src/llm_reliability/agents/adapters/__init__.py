"""
LLM Agent Adapter Framework public API.
"""

from __future__ import annotations

from llm_reliability.agents.adapters.base_llm_adapter import BaseLLMAdapter
from llm_reliability.agents.adapters.exceptions import (
    AuthenticationError,
    ConnectionError,
    ProviderError,
    RateLimitError,
    RequestValidationError,
    ResponseValidationError,
)
from llm_reliability.agents.adapters.provider_registry import ProviderRegistry
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse

__all__ = [
    "BaseLLMAdapter",
    "ProviderRegistry",
    "LLMRequest",
    "LLMResponse",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "RequestValidationError",
    "ResponseValidationError",
    "ConnectionError",
]
