from __future__ import annotations

import logging
import os
import time
from typing import Any

from llm_reliability.agents.adapters.base_llm_adapter import BaseLLMAdapter
from llm_reliability.agents.adapters.exceptions import (
    AuthenticationError,
)
from llm_reliability.agents.adapters.exceptions import (
    ConnectionError as ProviderConnectionError,
)
from llm_reliability.agents.adapters.exceptions import (
    ProviderError,
    RateLimitError,
    ResponseValidationError,
)
from llm_reliability.agents.adapters.provider_registry import ProviderRegistry
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse
from llm_reliability.configs.config import Configuration
from llm_reliability.runtime.provider_base import BaseProvider
from llm_reliability.runtime.registry import RuntimeRegistry

logger = logging.getLogger(__name__)


class _AzureOpenAIAdapter(BaseLLMAdapter):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._client = None
        self._model = config.metadata.get("model", config.llm) or "gpt-4o"
        self._temperature = float(config.metadata.get("temperature", 0.0))
        self._max_tokens = int(config.metadata.get("max_tokens", 1024))
        self._top_p = float(config.metadata.get("top_p", 1.0))

    def initialize(self) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise ImportError("The 'openai' package is required for Azure OpenAI.") from exc
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not api_key:
            raise AuthenticationError("AZURE_OPENAI_API_KEY environment variable is not set.")
        if not endpoint:
            raise AuthenticationError("AZURE_OPENAI_ENDPOINT environment variable is not set.")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        logger.info(
            "Azure OpenAI client initialised (model=%s, endpoint=%s).", self._model, endpoint
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("_AzureOpenAIAdapter.generate() called before initialize().")
        try:
            from openai import APIConnectionError as OpenAIConnectionError
            from openai import APIError as OpenAIAPIError
            from openai import AuthenticationError as OpenAIAuthError
            from openai import RateLimitError as OpenAIRateLimitError
        except ImportError as exc:
            raise ProviderError("openai package not available.") from exc
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        t0 = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
            )
        except OpenAIAuthError as exc:
            raise AuthenticationError(f"Azure OpenAI auth failed: {exc}") from exc
        except OpenAIRateLimitError as exc:
            raise RateLimitError(f"Azure OpenAI rate limit: {exc}") from exc
        except OpenAIConnectionError as exc:
            raise ProviderConnectionError(f"Azure OpenAI network error: {exc}") from exc
        except OpenAIAPIError as exc:
            raise ProviderError(f"Azure OpenAI API error: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0
        choice = completion.choices[0] if completion.choices else None
        if choice is None or not getattr(choice.message, "content", None):
            raise ResponseValidationError("Azure OpenAI returned empty completion.")
        text = choice.message.content or ""
        usage = completion.usage
        return LLMResponse(
            text=text,
            finish_reason=choice.finish_reason or "unknown",
            latency_ms=latency_ms,
            tokens_input=usage.prompt_tokens if usage else 0,
            tokens_output=usage.completion_tokens if usage else 0,
            model_name=completion.model or self._model,
            provider="azure_openai",
        )

    def shutdown(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def provider_metadata(self) -> dict[str, Any]:
        return {"provider": "azure_openai", "model": self._model}

    def health_check(self) -> bool:
        return self._client is not None


class AzureOpenAIAgent(BaseProvider):
    provider_name: str = "azure_openai"
    default_model: str = "gpt-4"
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    default_requests_per_second: float = 5.0
    api_key_env: str = "AZURE_OPENAI_API_KEY"

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._adapter = _AzureOpenAIAdapter(config)

    def initialize(self) -> None:
        self._adapter.initialize()
        self._client = getattr(self._adapter, "_client", None)

    def reset(self) -> None:
        super().reset()
        self._adapter._request_logs.clear()
        self._adapter._response_logs.clear()

    def run(self, task: dict[str, Any]) -> Any:
        prompt = self._extract_prompt(task)
        request = self._build_request(prompt)
        self._rate_limiter.acquire()
        response = self._adapter.retry(
            request, max_attempts=self._max_retries, backoff_seconds=self._retry_backoff
        )
        self._track_cost(response)
        return response.text

    def shutdown(self) -> None:
        self._adapter.shutdown()

    def metadata(self) -> dict[str, Any]:
        base = super().metadata()
        base.update(
            {"name": "AzureOpenAIAgent", "provider": "azure_openai", "model": self._adapter._model}
        )
        return base

    def _health_check_impl(self) -> bool:
        return self._adapter.health_check()


if not ProviderRegistry.exists("azure_openai"):
    ProviderRegistry.register("azure_openai", _AzureOpenAIAdapter)
if not RuntimeRegistry.exists("azure_openai"):
    RuntimeRegistry.register("azure_openai", AzureOpenAIAgent)
