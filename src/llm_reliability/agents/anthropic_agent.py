"""
AnthropicAgent — Claude provider adapter for LLM Reliability Ranking.

Wraps the ``anthropic`` SDK (Messages API) using the same two-layer
architecture as ``GPTAgent``:

    AnthropicAgent(Agent)
      └── _AnthropicAdapter(BaseLLMAdapter)

Environment variables
---------------------
ANTHROPIC_API_KEY   Required — Anthropic secret key.
ANTHROPIC_BASE_URL  Optional — Override base URL.

Configuration metadata keys (all optional)
------------------------------------------
model               str   — Default: ``"claude-3-5-sonnet-20241022"``
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
from llm_reliability.agents.adapters.exceptions import AuthenticationError
from llm_reliability.agents.adapters.exceptions import \
    ConnectionError as ProviderConnectionError
from llm_reliability.agents.adapters.exceptions import (
    ProviderError, RateLimitError, ResponseValidationError)
from llm_reliability.agents.adapters.provider_registry import ProviderRegistry
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse
from llm_reliability.configs.config import Configuration
from llm_reliability.runtime.provider_base import BaseProvider
from llm_reliability.runtime.registry import RuntimeRegistry

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "claude-3-5-sonnet-20241022"
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_REQUESTS_PER_SECOND: float = 2.0

ANTHROPIC_AGENT_VERSION: str = "1.0"


class _AnthropicAdapter(BaseLLMAdapter):
    """Internal Anthropic Messages API adapter."""

    _client: Any

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._client = None
        self._model = config.metadata.get("model", config.llm) or DEFAULT_MODEL
        # Anthropic models use "claude-" prefix; normalise if user gave short name
        if self._model and not self._model.startswith("claude"):
            self._model = DEFAULT_MODEL
        self._temperature = float(config.metadata.get("temperature", DEFAULT_TEMPERATURE))
        self._max_tokens = int(config.metadata.get("max_tokens", DEFAULT_MAX_TOKENS))
        self._system_prompt: str | None = config.metadata.get("system_prompt")

    def initialize(self) -> None:
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required. Install with: pip install anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise AuthenticationError("ANTHROPIC_API_KEY environment variable is not set.")

        kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url

        self._client = anthropic.Anthropic(**kwargs)
        logger.info(
            "Anthropic client initialised (model=%s, temperature=%.2f, max_tokens=%d).",
            self._model,
            self._temperature,
            self._max_tokens,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("_AnthropicAdapter.generate() called before initialize().")

        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("anthropic package not available.") from exc

        messages = [{"role": "user", "content": request.prompt}]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        # system prompt is a top-level param in Anthropic API
        system = request.system_prompt or self._system_prompt
        if system:
            kwargs["system"] = system

        t0 = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise AuthenticationError(f"Authentication failed: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise RateLimitError(f"Rate limit exceeded: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderConnectionError(f"Network error: {exc}") from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderConnectionError(f"Request timed out: {exc}") from exc
        except anthropic.BadRequestError as exc:
            raise ProviderError(f"Bad request: {exc}") from exc
        except anthropic.APIError as exc:
            raise ProviderError(f"API error: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not response.content or not response.content[0].text:
            raise ResponseValidationError(f"Anthropic returned empty content. Response: {response}")

        text = response.content[0].text
        usage = response.usage

        return LLMResponse(
            text=text,
            finish_reason=response.stop_reason or "unknown",
            latency_ms=latency_ms,
            tokens_input=usage.input_tokens if usage else 0,
            tokens_output=usage.output_tokens if usage else 0,
            model_name=response.model or self._model,
            provider="anthropic",
            metadata={"id": response.id},
        )

    def shutdown(self) -> None:
        self._client = None
        logger.debug("Anthropic client shut down.")

    def provider_metadata(self) -> dict[str, Any]:
        return {
            "provider": "anthropic",
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            # Minimal check: list models endpoint
            self._client.models.list()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Anthropic health check failed: %s", exc)
            return False


class AnthropicAgent(BaseProvider):
    """Claude agent for the LLM Reliability Ranking framework."""

    provider_name: str = "anthropic"
    default_model: str = "claude-3-5-sonnet-20241022"
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    default_requests_per_second: float = 2.0
    api_key_env: str = "ANTHROPIC_API_KEY"
    api_base_env: str = "ANTHROPIC_BASE_URL"

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._adapter = _AnthropicAdapter(config)

    def initialize(self) -> None:
        logger.info("Initialising AnthropicAgent.")
        self._adapter.initialize()
        self._client = getattr(self._adapter, "_client", None)
        logger.info("AnthropicAgent ready (model=%s).", self._adapter._model)

    def reset(self) -> None:
        super().reset()
        self._adapter._request_logs.clear()
        self._adapter._response_logs.clear()

    def run(self, task: dict[str, Any]) -> Any:
        prompt = self._extract_prompt(task)
        request = LLMRequest(
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            seed=None,
            system_prompt=self._system_prompt,
        )
        logger.info(
            "AnthropicAgent.run: task_id=%r, model=%s, prompt_len=%d.",
            task.get("task_id", "<unknown>"),
            self._adapter._model,
            len(prompt),
        )
        self._rate_limiter.acquire()
        response = self._adapter.retry(
            request, max_attempts=self._max_retries, backoff_seconds=self._retry_backoff
        )
        self._track_cost(response)
        logger.info(
            "AnthropicAgent.run complete: task_id=%r, finish=%s.",
            task.get("task_id", "<unknown>"),
            response.finish_reason,
        )
        return response.text

    def shutdown(self) -> None:
        logger.info("Shutting down AnthropicAgent.")
        self._adapter.shutdown()

    def metadata(self) -> dict[str, Any]:
        base = super().metadata()
        base.update(
            {
                "name": "AnthropicAgent",
                "provider": "anthropic",
                "model": self._adapter._model,
                "version": ANTHROPIC_AGENT_VERSION,
            }
        )
        return base

    def _health_check_impl(self) -> bool:
        return self._adapter.health_check()


if not ProviderRegistry.exists("anthropic"):
    ProviderRegistry.register("anthropic", _AnthropicAdapter)
if not RuntimeRegistry.exists("anthropic"):
    RuntimeRegistry.register("anthropic", AnthropicAgent)
