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

DEFAULT_MODEL = "llama3.1-8b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024


class _CerebrasAdapter(BaseLLMAdapter):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._client = None
        self._model = config.metadata.get("model", config.llm) or DEFAULT_MODEL
        self._temperature = float(config.metadata.get("temperature", DEFAULT_TEMPERATURE))
        self._max_tokens = int(config.metadata.get("max_tokens", DEFAULT_MAX_TOKENS))

    def initialize(self) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError("The 'openai' package is required for Cerebras.") from exc
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise AuthenticationError("CEREBRAS_API_KEY environment variable is not set.")
        self._client = openai.OpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=api_key,
        )
        logger.info("Cerebras client initialised (model=%s).", self._model)

    def generate(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("_CerebrasAdapter.generate() called before initialize().")
        try:
            import openai
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
            )
        except openai.AuthenticationError as exc:
            raise AuthenticationError(f"Cerebras auth failed: {exc}") from exc
        except openai.RateLimitError as exc:
            raise RateLimitError(f"Cerebras rate limit: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise ProviderConnectionError(f"Cerebras network error: {exc}") from exc
        except openai.APIError as exc:
            raise ProviderError(f"Cerebras API error: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0
        choice = completion.choices[0] if completion.choices else None
        if choice is None or not getattr(choice.message, "content", None):
            raise ResponseValidationError("Cerebras returned empty completion.")
        text = choice.message.content or ""
        usage = completion.usage
        return LLMResponse(
            text=text,
            finish_reason=choice.finish_reason or "unknown",
            latency_ms=latency_ms,
            tokens_input=usage.prompt_tokens if usage else 0,
            tokens_output=usage.completion_tokens if usage else 0,
            model_name=completion.model or self._model,
            provider="cerebras",
        )

    def shutdown(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def provider_metadata(self) -> dict[str, Any]:
        return {"provider": "cerebras", "model": self._model}

    def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.models.list()
            return True
        except Exception:
            return False


class CerebrasAgent(BaseProvider):
    provider_name: str = "cerebras"
    default_model: str = "llama3.1-8b"
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    default_requests_per_second: float = 5.0
    api_key_env: str = "CEREBRAS_API_KEY"

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._adapter = _CerebrasAdapter(config)

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
            {"name": "CerebrasAgent", "provider": "cerebras", "model": self._adapter._model}
        )
        return base

    def _health_check_impl(self) -> bool:
        return self._adapter.health_check()


if not ProviderRegistry.exists("cerebras"):
    ProviderRegistry.register("cerebras", _CerebrasAdapter)
if not RuntimeRegistry.exists("cerebras"):
    RuntimeRegistry.register("cerebras", CerebrasAgent)
