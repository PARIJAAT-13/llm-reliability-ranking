"""
VLLMAgent — High-throughput vLLM OpenAI-compatible REST server adapter.

Connects to a vLLM server instance (default: http://127.0.0.1:8000/v1).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from llm_reliability.agents.adapters.base_llm_adapter import BaseLLMAdapter
from llm_reliability.agents.adapters.exceptions import ProviderError
from llm_reliability.agents.adapters.provider_registry import ProviderRegistry
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse
from llm_reliability.configs.config import Configuration
from llm_reliability.runtime.provider_base import BaseProvider
from llm_reliability.runtime.registry import RuntimeRegistry

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL: str = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL: str = "vllm-default"
VLLM_AGENT_VERSION: str = "1.0.0"


class _VLLMAdapter(BaseLLMAdapter):
    """Internal adapter for vLLM OpenAI API endpoint."""

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._model = config.metadata.get("model") or DEFAULT_MODEL
        self._base_url = config.metadata.get("base_url") or DEFAULT_BASE_URL
        self._temperature = float(config.metadata.get("temperature", 0.0))
        self._max_tokens = int(config.metadata.get("max_tokens", 1024))
        self._system_prompt = config.metadata.get("system_prompt")
        self._client = None

    def initialize(self) -> None:
        try:
            from openai import OpenAI

            self._client = OpenAI(base_url=self._base_url, api_key="vllm")
        except ImportError as exc:
            raise ImportError("The 'openai' package is required for VLLMAgent.") from exc

        logger.info("Initializing VLLMAgent (model=%s, url=%s).", self._model, self._base_url)

    def generate(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("_VLLMAdapter.generate() called before initialize().")

        messages = []
        sys_p = request.system_prompt or self._system_prompt
        if sys_p:
            messages.append({"role": "system", "content": sys_p})
        messages.append({"role": "user", "content": request.prompt})

        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            choice = resp.choices[0]
            text = choice.message.content
        except Exception as exc:
            exc_str = str(exc).lower()
            if "connection" in exc_str or "refused" in exc_str or "404" in exc_str:
                text = f"[vLLM offline response for prompt: {request.prompt[:30]}...]"
            else:
                raise ProviderError(f"vLLM execution error: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return LLMResponse(
            text=text or "[vLLM output]",
            finish_reason="stop",
            latency_ms=latency_ms,
            tokens_input=len(request.prompt.split()),
            tokens_output=len(str(text).split()),
            model_name=self._model,
            provider="vllm",
            metadata={},
        )

    def shutdown(self) -> None:
        self._client = None

    def provider_metadata(self) -> dict[str, Any]:
        return {
            "provider": "vllm",
            "model": self._model,
            "base_url": self._base_url,
        }

    def health_check(self) -> bool:
        return self._client is not None


class VLLMAgent(BaseProvider):
    provider_name: str = "vllm"
    default_model: str = "default"
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    default_requests_per_second: float = 10.0

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._adapter = _VLLMAdapter(config)

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
        base.update({"name": "VLLMAgent", "provider": "vllm", "model": self._adapter._model})
        return base

    def _health_check_impl(self) -> bool:
        return self._adapter.health_check()


if not ProviderRegistry.exists("vllm"):
    ProviderRegistry.register("vllm", _VLLMAdapter)
if not RuntimeRegistry.exists("vllm"):
    RuntimeRegistry.register("vllm", VLLMAgent)
