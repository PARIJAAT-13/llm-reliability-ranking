from __future__ import annotations

import logging
import os
import time
from typing import Any

from llm_reliability.agents.adapters.base_llm_adapter import BaseLLMAdapter
from llm_reliability.agents.adapters.exceptions import (
    AuthenticationError,
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


def _check_litellm() -> None:
    try:
        import litellm  # noqa: F401
    except ImportError as exc:
        raise ImportError("The 'litellm' package is required for LiteLLMAgent.") from exc


class _LiteLLMAdapter(BaseLLMAdapter):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._model = config.metadata.get("model", config.llm) or "gpt-4o"
        self._temperature = float(config.metadata.get("temperature", 0.0))
        self._max_tokens = int(config.metadata.get("max_tokens", 1024))
        self._top_p = float(config.metadata.get("top_p", 1.0))

    def initialize(self) -> None:
        _check_litellm()
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("LiteLLM: No provider API keys found in environment.")
        logger.info("LiteLLM client initialised (model=%s).", self._model)

    def generate(self, request: LLMRequest) -> LLMResponse:
        import litellm

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        t0 = time.perf_counter()
        try:
            response = litellm.completion(
                model=self._model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "auth" in exc_str or "401" in exc_str:
                raise AuthenticationError(f"LiteLLM auth failed: {exc}") from exc
            elif "rate" in exc_str or "429" in exc_str:
                raise RateLimitError(f"LiteLLM rate limit: {exc}") from exc
            raise ProviderError(f"LiteLLM API error: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0
        choice = response.choices[0] if response.choices else None
        if choice is None or not getattr(choice.message, "content", None):
            raise ResponseValidationError("LiteLLM returned empty completion.")
        text = choice.message.content or ""
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            finish_reason=choice.finish_reason or "stop",
            latency_ms=latency_ms,
            tokens_input=usage.prompt_tokens if usage else 0,
            tokens_output=usage.completion_tokens if usage else 0,
            model_name=response.model or self._model,
            provider="litellm",
        )

    def shutdown(self) -> None:
        pass

    def provider_metadata(self) -> dict[str, Any]:
        return {"provider": "litellm", "model": self._model}

    def health_check(self) -> bool:
        return True


class LiteLLMAgent(BaseProvider):
    provider_name: str = "litellm"
    default_model: str = "gpt-3.5-turbo"
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    default_requests_per_second: float = 5.0

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._adapter = _LiteLLMAdapter(config)

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
        base.update({"name": "LiteLLMAgent", "provider": "litellm", "model": self._adapter._model})
        return base

    def _health_check_impl(self) -> bool:
        return self._adapter.health_check()


if not ProviderRegistry.exists("litellm"):
    ProviderRegistry.register("litellm", _LiteLLMAdapter)
if not RuntimeRegistry.exists("litellm"):
    RuntimeRegistry.register("litellm", LiteLLMAgent)
