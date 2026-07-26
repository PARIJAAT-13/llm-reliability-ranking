from __future__ import annotations

import logging
import os
import time
from typing import Any

from llm_reliability.agents.adapters.base_llm_adapter import BaseLLMAdapter
from llm_reliability.agents.adapters.exceptions import (
    AuthenticationError, ProviderError, RateLimitError,
    ResponseValidationError)
from llm_reliability.agents.adapters.provider_registry import ProviderRegistry
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse
from llm_reliability.configs.config import Configuration
from llm_reliability.runtime.provider_base import BaseProvider
from llm_reliability.runtime.registry import RuntimeRegistry

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mistral-large-2407"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024


class _MistralAdapter(BaseLLMAdapter):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._client = None
        self._model = config.metadata.get("model", config.llm) or DEFAULT_MODEL
        self._temperature = float(config.metadata.get("temperature", DEFAULT_TEMPERATURE))
        self._max_tokens = int(config.metadata.get("max_tokens", DEFAULT_MAX_TOKENS))
        self._top_p = float(config.metadata.get("top_p", 1.0))

    def initialize(self) -> None:
        try:
            from mistralai import Mistral
        except ImportError as exc:
            raise ImportError("The 'mistralai' package is required for MistralAgent.") from exc
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise AuthenticationError("MISTRAL_API_KEY environment variable is not set.")
        self._client = Mistral(api_key=api_key)
        logger.info("Mistral client initialised (model=%s).", self._model)

    def generate(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("_MistralAdapter.generate() called before initialize().")

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        t0 = time.perf_counter()
        try:
            response = self._client.chat.complete(
                model=self._model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "unauthorized" in exc_str or "auth" in exc_str:
                raise AuthenticationError(f"Mistral auth failed: {exc}") from exc
            elif "rate" in exc_str or "429" in exc_str:
                raise RateLimitError(f"Mistral rate limit: {exc}") from exc
            raise ProviderError(f"Mistral API error: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        choice = response.choices[0] if response.choices else None
        if choice is None or not getattr(choice.message, "content", None):
            raise ResponseValidationError("Mistral returned empty completion.")
        text = choice.message.content or ""
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            finish_reason=choice.finish_reason or "stop",
            latency_ms=latency_ms,
            tokens_input=usage.prompt_tokens if usage else 0,
            tokens_output=usage.completion_tokens if usage else 0,
            model_name=self._model,
            provider="mistral",
        )

    def shutdown(self) -> None:
        self._client = None

    def provider_metadata(self) -> dict[str, Any]:
        return {"provider": "mistral", "model": self._model}

    def health_check(self) -> bool:
        return self._client is not None


class MistralAgent(BaseProvider):
    provider_name: str = "mistral"
    default_model: str = "mistral-large-2407"
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    default_top_p: float = 1.0
    default_requests_per_second: float = 5.0
    api_key_env: str = "MISTRAL_API_KEY"

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._adapter = _MistralAdapter(config)

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
            {
                "name": "MistralAgent",
                "provider": "mistral",
                "model": self._adapter._model,
            }
        )
        return base

    def _health_check_impl(self) -> bool:
        return self._adapter.health_check()


if not ProviderRegistry.exists("mistral"):
    ProviderRegistry.register("mistral", _MistralAdapter)
if not RuntimeRegistry.exists("mistral"):
    RuntimeRegistry.register("mistral", MistralAgent)
