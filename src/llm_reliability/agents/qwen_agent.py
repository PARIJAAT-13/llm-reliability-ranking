"""
QwenAgent — Alibaba Qwen provider adapter for LLM Reliability Ranking.

Qwen models are accessed via Alibaba Cloud's DashScope API, which exposes
an OpenAI-compatible endpoint. This adapter uses the ``openai`` SDK with
``base_url`` pointed at the DashScope endpoint.

    QwenAgent(Agent)
      └── _QwenAdapter(BaseLLMAdapter)

Environment variables
---------------------
QWEN_API_KEY        Required — DashScope API key (also exported as DASHSCOPE_API_KEY).
QWEN_BASE_URL       Optional — Override base URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1).

Configuration metadata keys (all optional)
------------------------------------------
model               str   — Default: ``"qwen-2.5-72b-instruct"``
temperature         float — Default: ``0.0``
max_tokens          int   — Default: ``1024``
system_prompt       str   — Optional system-level instructions.
max_retries         int   — Default: ``3``
retry_backoff       float — Default: ``1.0``
requests_per_second float — Default: ``2.0``
"""

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

QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_AGENT_VERSION: str = "1.0"


class _QwenAdapter(BaseLLMAdapter):
    """Internal Qwen adapter (DashScope OpenAI-compatible API)."""

    _client: Any

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._client = None
        self._model = config.metadata.get("model", config.llm) or "qwen-2.5-72b-instruct"
        if self._model and not self._model.startswith("qwen"):
            self._model = "qwen-2.5-72b-instruct"
        self._temperature = float(config.metadata.get("temperature", 0.0))
        self._max_tokens = int(config.metadata.get("max_tokens", 1024))
        self._system_prompt: str | None = config.metadata.get("system_prompt")

    def initialize(self) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for QwenAgent (DashScope OpenAI-compat API). "
                "Install with: pip install openai>=1.0"
            ) from exc

        api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "QWEN_API_KEY (or DASHSCOPE_API_KEY) environment variable is not set."
            )

        base_url = os.environ.get("QWEN_BASE_URL", QWEN_BASE_URL)
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        logger.info(
            "Qwen client initialised (model=%s, base_url=%s, temperature=%.2f, max_tokens=%d).",
            self._model,
            base_url,
            self._temperature,
            self._max_tokens,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("_QwenAdapter.generate() called before initialize().")

        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("openai package not available.") from exc

        messages: list[dict[str, str]] = []
        system = request.system_prompt or self._system_prompt
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": request.prompt})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        t0 = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as exc:
            raise AuthenticationError(f"Authentication failed: {exc}") from exc
        except openai.RateLimitError as exc:
            raise RateLimitError(f"Rate limit exceeded: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise ProviderConnectionError(f"Network error: {exc}") from exc
        except openai.APITimeoutError as exc:
            raise ProviderConnectionError(f"Request timed out: {exc}") from exc
        except openai.APIError as exc:
            raise ProviderError(f"API error: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        choice = completion.choices[0] if completion.choices else None
        if choice is None or not getattr(choice.message, "content", None):
            raise ResponseValidationError(f"Qwen returned empty completion. Response: {completion}")

        text = choice.message.content or ""
        usage = completion.usage

        return LLMResponse(
            text=text,
            finish_reason=choice.finish_reason or "unknown",
            latency_ms=latency_ms,
            tokens_input=usage.prompt_tokens if usage else 0,
            tokens_output=usage.completion_tokens if usage else 0,
            model_name=completion.model or self._model,
            provider="qwen",
            metadata={"id": completion.id},
        )

    def shutdown(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        logger.debug("Qwen client shut down.")

    def provider_metadata(self) -> dict[str, Any]:
        return {
            "provider": "qwen",
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.models.list()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qwen health check failed: %s", exc)
            return False


class QwenAgent(BaseProvider):
    provider_name: str = "qwen"
    default_model: str = "qwen2.5-72b-instruct"
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    default_requests_per_second: float = 2.0
    api_key_env: str = "QWEN_API_KEY"

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._adapter = _QwenAdapter(config)

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
        base.update({"name": "QwenAgent", "provider": "qwen", "model": self._adapter._model})
        return base

    def _health_check_impl(self) -> bool:
        return self._adapter.health_check()


if not ProviderRegistry.exists("qwen"):
    ProviderRegistry.register("qwen", _QwenAdapter)
if not RuntimeRegistry.exists("qwen"):
    RuntimeRegistry.register("qwen", QwenAgent)
