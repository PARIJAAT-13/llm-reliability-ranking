"""
GeminiAgent — Google Gemini provider adapter for LLM Reliability Ranking.

Wraps the ``google-generativeai`` SDK using the same two-layer architecture:

    GeminiAgent(Agent)
      └── _GeminiAdapter(BaseLLMAdapter)

Environment variables
---------------------
GEMINI_API_KEY      Required — Google AI Studio / Vertex API key.

Configuration metadata keys (all optional)
------------------------------------------
model               str   — Default: ``"gemini-1.5-pro"``
temperature         float — Default: ``0.0``
max_tokens          int   — Default: ``1024``  (maps to max_output_tokens)
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
    ConnectionError as ProviderConnectionError,
    ProviderError,
    RateLimitError,
    ResponseValidationError,
)
from llm_reliability.agents.adapters.provider_registry import ProviderRegistry
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse
from llm_reliability.agents.utils.rate_limiter import RateLimiter
from llm_reliability.configs.config import Configuration
from llm_reliability.interfaces.agent import Agent

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "gemini-1.5-pro"
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_REQUESTS_PER_SECOND: float = 2.0
GEMINI_AGENT_VERSION: str = "1.0"

_PROMPT_KEYS: tuple[str, ...] = ("prompt", "question", "problem_statement")


class _GeminiAdapter(BaseLLMAdapter):
    """Internal Google Gemini generate-content adapter."""

    _client: Any  # google.generativeai.GenerativeModel

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._client = None
        self._model_name = config.metadata.get("model", config.llm) or DEFAULT_MODEL
        if self._model_name and not self._model_name.startswith("gemini"):
            self._model_name = DEFAULT_MODEL
        self._temperature = float(config.metadata.get("temperature", DEFAULT_TEMPERATURE))
        self._max_tokens = int(config.metadata.get("max_tokens", DEFAULT_MAX_TOKENS))
        self._system_prompt: str | None = config.metadata.get("system_prompt")

    def initialize(self) -> None:
        try:
            import google.generativeai as genai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'google-generativeai' package is required. "
                "Install with: pip install google-generativeai"
            ) from exc

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise AuthenticationError("GEMINI_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)
        generation_config = genai.GenerationConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
        )
        self._client = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config=generation_config,
            system_instruction=self._system_prompt,
        )
        self._genai = genai
        logger.info(
            "Gemini client initialised (model=%s, temperature=%.2f, max_tokens=%d).",
            self._model_name, self._temperature, self._max_tokens,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("_GeminiAdapter.generate() called before initialize().")

        prompt = request.prompt
        if request.system_prompt:
            # Prepend system prompt as a human turn prefix (Gemini style)
            prompt = f"{request.system_prompt}\n\n{prompt}"

        t0 = time.perf_counter()
        try:
            result = self._client.generate_content(prompt)
        except Exception as exc:  # noqa: BLE001
            exc_name = type(exc).__name__.lower()
            if "quota" in exc_name or "ratelimit" in exc_name or "429" in str(exc):
                raise RateLimitError(f"Rate limit exceeded: {exc}") from exc
            if "auth" in exc_name or "credential" in exc_name or "403" in str(exc):
                raise AuthenticationError(f"Authentication failed: {exc}") from exc
            if "connection" in exc_name or "timeout" in exc_name:
                raise ProviderConnectionError(f"Network error: {exc}") from exc
            raise ProviderError(f"Gemini API error: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        try:
            text = result.text
        except (ValueError, AttributeError) as exc:
            raise ResponseValidationError(
                f"Gemini returned no valid text. Result: {result!r}. Error: {exc}"
            ) from exc

        if not text or not text.strip():
            raise ResponseValidationError("Gemini returned an empty response.")

        usage = getattr(result, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", 0) if usage else 0
        tokens_out = getattr(usage, "candidates_token_count", 0) if usage else 0

        finish = "stop"
        if result.candidates:
            fr = getattr(result.candidates[0], "finish_reason", None)
            if fr is not None:
                finish = str(fr)

        return LLMResponse(
            text=text,
            finish_reason=finish,
            latency_ms=latency_ms,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            model_name=self._model_name,
            provider="google",
            metadata={},
        )

    def shutdown(self) -> None:
        self._client = None
        logger.debug("Gemini client shut down.")

    def provider_metadata(self) -> dict[str, Any]:
        return {
            "provider": "google",
            "model": self._model_name,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.generate_content("ping")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini health check failed: %s", exc)
            return False


class GeminiAgent(Agent):
    """Google Gemini agent for the LLM Reliability Ranking framework."""

    def __init__(self, config: Configuration) -> None:
        if config is None:
            raise ValueError("Configuration must be provided to GeminiAgent.")
        self._config = config
        self._adapter = _GeminiAdapter(config)
        self._max_retries: int = int(config.metadata.get("max_retries", 3))
        self._retry_backoff: float = float(config.metadata.get("retry_backoff", 1.0))
        self._rate_limiter = RateLimiter(
            requests_per_second=float(
                config.metadata.get("requests_per_second", DEFAULT_REQUESTS_PER_SECOND)
            )
        )

    def initialize(self) -> None:
        logger.info("Initialising GeminiAgent.")
        self._adapter.initialize()
        logger.info("GeminiAgent ready (model=%s).", self._adapter._model_name)

    def reset(self) -> None:
        self._adapter._request_logs.clear()
        self._adapter._response_logs.clear()

    def run(self, task: dict[str, Any]) -> Any:
        prompt = self._extract_prompt(task)
        request = LLMRequest(
            prompt=prompt,
            temperature=float(self._config.metadata.get("temperature", DEFAULT_TEMPERATURE)),
            max_tokens=int(self._config.metadata.get("max_tokens", DEFAULT_MAX_TOKENS)),
            system_prompt=self._config.metadata.get("system_prompt"),
        )
        logger.info("GeminiAgent.run: task_id=%r, prompt_len=%d.",
                    task.get("task_id", "<unknown>"), len(prompt))
        self._rate_limiter.acquire()
        response = self._adapter.retry(request, max_attempts=self._max_retries,
                                       backoff_seconds=self._retry_backoff)
        logger.info("GeminiAgent.run complete: task_id=%r, finish=%s.",
                    task.get("task_id", "<unknown>"), response.finish_reason)
        return response.text

    def shutdown(self) -> None:
        logger.info("Shutting down GeminiAgent.")
        self._adapter.shutdown()

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "GeminiAgent",
            "provider": "google",
            "model": self._adapter._model_name,
            "version": GEMINI_AGENT_VERSION,
            "temperature": self._adapter._temperature,
            "max_tokens": self._adapter._max_tokens,
            "seed": self._config.seed,
            "max_retries": self._max_retries,
        }

    @staticmethod
    def _extract_prompt(task: dict[str, Any]) -> str:
        for key in _PROMPT_KEYS:
            value = task.get(key)
            if value and str(value).strip():
                return str(value).strip()
        fallback = str(task).strip()
        if not fallback or fallback == "{}":
            raise ValueError(f"Cannot extract a prompt from task dict: {task!r}.")
        return fallback


if not ProviderRegistry.exists("google"):
    ProviderRegistry.register("google", _GeminiAdapter)
