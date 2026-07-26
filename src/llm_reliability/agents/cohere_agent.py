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

DEFAULT_MODEL = "command-r-plus"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024


class _CohereAdapter(BaseLLMAdapter):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._client = None
        self._model = config.metadata.get("model", config.llm) or DEFAULT_MODEL
        self._temperature = float(config.metadata.get("temperature", DEFAULT_TEMPERATURE))
        self._max_tokens = int(config.metadata.get("max_tokens", DEFAULT_MAX_TOKENS))

    def initialize(self) -> None:
        try:
            import cohere
        except ImportError as exc:
            raise ImportError("The 'cohere' package is required for CohereAgent.") from exc
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise AuthenticationError("COHERE_API_KEY environment variable is not set.")
        self._client = cohere.Client(api_key=api_key)
        logger.info("Cohere client initialised (model=%s).", self._model)

    def generate(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("_CohereAdapter.generate() called before initialize().")
        try:
            import cohere
        except ImportError as exc:
            raise ProviderError("cohere package not available.") from exc
        from cohere.core.api_error import ApiError

        t0 = time.perf_counter()
        try:
            response = self._client.chat(
                model=self._model,
                message=request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except cohere.UnauthorizedError as exc:
            raise AuthenticationError(f"Cohere auth failed: {exc}") from exc
        except cohere.TooManyRequestsError as exc:
            raise RateLimitError(f"Cohere rate limit: {exc}") from exc
        except ApiError as exc:
            raise ProviderError(f"Cohere API error: {exc}") from exc
        except Exception as exc:
            raise ProviderConnectionError(f"Cohere connection error: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = getattr(response, "text", "") or ""
        if not text.strip():
            raise ResponseValidationError("Cohere returned empty response.")
        usage = getattr(response, "token_count", None)
        return LLMResponse(
            text=text,
            finish_reason="stop",
            latency_ms=latency_ms,
            tokens_input=getattr(usage, "input_tokens", 0) if usage else 0,
            tokens_output=getattr(usage, "output_tokens", 0) if usage else 0,
            model_name=self._model,
            provider="cohere",
        )

    def shutdown(self) -> None:
        self._client = None

    def provider_metadata(self) -> dict[str, Any]:
        return {"provider": "cohere", "model": self._model}

    def health_check(self) -> bool:
        return self._client is not None


class CohereAgent(BaseProvider):
    provider_name: str = "cohere"
    default_model: str = "command-r-plus"
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    default_requests_per_second: float = 5.0
    api_key_env: str = "COHERE_API_KEY"

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._adapter = _CohereAdapter(config)

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
        base.update({"name": "CohereAgent", "provider": "cohere", "model": self._adapter._model})
        return base

    def _health_check_impl(self) -> bool:
        return self._adapter.health_check()


if not ProviderRegistry.exists("cohere"):
    ProviderRegistry.register("cohere", _CohereAdapter)
if not RuntimeRegistry.exists("cohere"):
    RuntimeRegistry.register("cohere", CohereAgent)
