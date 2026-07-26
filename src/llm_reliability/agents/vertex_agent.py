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


class _VertexAdapter(BaseLLMAdapter):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._client = None
        self._model = config.metadata.get("model", config.llm) or "gemini-2.5-pro-preview-03-25"
        self._temperature = float(config.metadata.get("temperature", 0.0))
        self._max_tokens = int(config.metadata.get("max_tokens", 1024))
        self._top_p = float(config.metadata.get("top_p", 1.0))

    def initialize(self) -> None:
        try:
            import google.cloud.aiplatform as aiplatform
            from vertexai.preview.generative_models import GenerativeModel
        except ImportError as exc:
            raise ImportError(
                "The 'google-cloud-aiplatform' package is required for Vertex AI."
            ) from exc
        project = os.environ.get("VERTEX_AI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
        if not project:
            raise AuthenticationError(
                "VERTEX_AI_PROJECT or GOOGLE_CLOUD_PROJECT environment variable must be set."
            )
        aiplatform.init(project=project, location=location)
        self._client = GenerativeModel(self._model)
        logger.info("Vertex AI client initialised (model=%s, project=%s).", self._model, project)

    def generate(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("_VertexAdapter.generate() called before initialize().")
        t0 = time.perf_counter()
        try:
            contents = [request.prompt]
            if request.system_prompt:
                contents.insert(0, f"[System] {request.system_prompt}")
            response = self._client.generate_content(
                contents,
                generation_config={
                    "temperature": request.temperature,
                    "max_output_tokens": request.max_tokens,
                    "top_p": request.top_p,
                },
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "unauthorized" in exc_str or "auth" in exc_str:
                raise AuthenticationError(f"Vertex AI auth failed: {exc}") from exc
            elif "quota" in exc_str or "rate" in exc_str:
                raise RateLimitError(f"Vertex AI rate limit: {exc}") from exc
            raise ProviderError(f"Vertex AI API error: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = response.text if hasattr(response, "text") else ""
        if not text.strip():
            raise ResponseValidationError("Vertex AI returned empty response.")
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=text,
            finish_reason="stop",
            latency_ms=latency_ms,
            tokens_input=usage.prompt_token_count if usage else 0,
            tokens_output=usage.candidates_token_count if usage else 0,
            model_name=self._model,
            provider="vertex",
        )

    def shutdown(self) -> None:
        self._client = None

    def provider_metadata(self) -> dict[str, Any]:
        return {"provider": "vertex", "model": self._model}

    def health_check(self) -> bool:
        return self._client is not None


class VertexAgent(BaseProvider):
    provider_name: str = "vertex"
    default_model: str = "gemini-1.5-pro"
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    default_requests_per_second: float = 5.0

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._adapter = _VertexAdapter(config)

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
        base.update({"name": "VertexAgent", "provider": "vertex", "model": self._adapter._model})
        return base

    def _health_check_impl(self) -> bool:
        return self._adapter.health_check()


if not ProviderRegistry.exists("vertex"):
    ProviderRegistry.register("vertex", _VertexAdapter)
if not RuntimeRegistry.exists("vertex"):
    RuntimeRegistry.register("vertex", VertexAgent)
