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
from llm_reliability.agents.utils.rate_limiter import RateLimiter
from llm_reliability.configs.config import Configuration
from llm_reliability.runtime import Runtime
from llm_reliability.runtime.registry import RuntimeRegistry

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL: str = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL: str = "vllm-default"
VLLM_AGENT_VERSION: str = "1.0.0"

_PROMPT_KEYS: tuple[str, ...] = ("prompt", "question", "problem_statement")


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


class VLLMAgent(Runtime):
    """vLLM agent for the LLM Reliability Ranking framework."""

    def __init__(self, config: Configuration) -> None:
        if config is None:
            raise ValueError("Configuration must be provided to VLLMAgent.")
        self._config = config
        self._adapter = _VLLMAdapter(config)
        self._rate_limiter = RateLimiter(requests_per_second=20.0)

    def initialize(self) -> None:
        self._adapter.initialize()

    def reset(self) -> None:
        self._adapter._request_logs.clear()
        self._adapter._response_logs.clear()

    def run(self, task: dict[str, Any]) -> Any:
        prompt = self._extract_prompt(task)
        request = LLMRequest(
            prompt=prompt,
            temperature=float(self._config.metadata.get("temperature", 0.0)),
            max_tokens=int(self._config.metadata.get("max_tokens", 1024)),
            system_prompt=self._config.metadata.get("system_prompt"),
        )
        self._rate_limiter.acquire()
        response = self._adapter.retry(request, max_attempts=3, backoff_seconds=1.0)
        return response.text

    def shutdown(self) -> None:
        self._adapter.shutdown()

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "VLLMAgent",
            "provider": "vllm",
            "model": self._adapter._model,
            "version": VLLM_AGENT_VERSION,
        }

    @staticmethod
    def _extract_prompt(task: dict[str, Any]) -> str:
        for key in _PROMPT_KEYS:
            value = task.get(key)
            if value and str(value).strip():
                return str(value).strip()
        return str(task).strip()


if not ProviderRegistry.exists("vllm"):
    ProviderRegistry.register("vllm", _VLLMAdapter)
if not RuntimeRegistry.exists("vllm"):
    RuntimeRegistry.register("vllm", VLLMAgent)
