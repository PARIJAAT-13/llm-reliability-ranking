"""
LlamaAgent — Meta Llama provider adapter for LLM Reliability Ranking.

Llama models are accessed via the HuggingFace Inference API (serverless or
dedicated endpoints).  The adapter uses the ``huggingface_hub`` InferenceClient,
which provides a chat_completion method compatible with the OpenAI Messages format.

    LlamaAgent(Agent)
      └── _LlamaAdapter(BaseLLMAdapter)

Environment variables
---------------------
HF_TOKEN            Required — HuggingFace access token.
LLAMA_BASE_URL      Optional — Override endpoint URL (for dedicated endpoints or
                               local inference servers like vLLM / TGI).

Configuration metadata keys (all optional)
------------------------------------------
model               str   — Default: ``"meta-llama/Llama-3.3-70B-Instruct"``
temperature         float — Default: ``0.0``  (maps to temperature in HF API)
max_tokens          int   — Default: ``1024``
system_prompt       str   — Optional system-level instructions.
max_retries         int   — Default: ``3``
retry_backoff       float — Default: ``1.0``
requests_per_second float — Default: ``1.0``

Note: For serverless HF inference the ``temperature`` must be > 0 to avoid
deterministic mode restrictions on some models.  Setting it to ``0.01`` is
a safe workaround; this adapter applies that floor automatically.
"""

from __future__ import annotations

import logging
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
from llm_reliability.agents.utils.rate_limiter import RateLimiter
from llm_reliability.configs.config import Configuration
from llm_reliability.runtime import Runtime
from llm_reliability.runtime.registry import RuntimeRegistry

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_TEMPERATURE: float = 0.01  # HF serverless requires > 0 for most Llama models
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_REQUESTS_PER_SECOND: float = 1.0  # HF serverless rate limits are conservative
LLAMA_AGENT_VERSION: str = "1.0"

_PROMPT_KEYS: tuple[str, ...] = ("prompt", "question", "problem_statement")

# Minimum temperature for HF serverless to avoid "greedy not supported" errors
_MIN_TEMPERATURE: float = 0.01


class _LlamaAdapter(BaseLLMAdapter):
    """Internal Llama adapter using HuggingFace InferenceClient."""

    _client: Any  # huggingface_hub.InferenceClient

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._client = None
        configured_model = config.metadata.get("model") or config.llm
        if configured_model == "llama":
            configured_model = "llama3.1:8b"
        self._model = configured_model
        # Accept short names like "llama-3.3-70b" and expand to full HF path

        raw_temp = float(config.metadata.get("temperature", DEFAULT_TEMPERATURE))
        # Apply minimum temperature floor for HF serverless compatibility
        self._temperature = max(raw_temp, _MIN_TEMPERATURE)
        self._max_tokens = int(config.metadata.get("max_tokens", DEFAULT_MAX_TOKENS))
        self._system_prompt: str | None = config.metadata.get("system_prompt")

    def initialize(self) -> None:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for LlamaAgent. "
                "Install with: pip install openai"
            ) from exc

        self._client = OpenAI(
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama",
        )

        logger.info(
            "Llama client initialised (model=%s, temperature=%.4f, max_tokens=%d).",
            self._model,
            self._temperature,
            self._max_tokens,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("_LlamaAdapter.generate() called before initialize().")

        messages: list[dict[str, str]] = []
        system = request.system_prompt or self._system_prompt
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": request.prompt})

        t0 = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                messages=messages,
                model=self._model,
                temperature=max(request.temperature, _MIN_TEMPERATURE),
                max_tokens=request.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            exc_str = str(exc).lower()
            if "unauthorized" in exc_str or "401" in exc_str:
                raise AuthenticationError(f"Authentication failed: {exc}") from exc
            if "rate limit" in exc_str or "429" in exc_str or "too many" in exc_str:
                raise RateLimitError(f"Rate limit exceeded: {exc}") from exc
            if "connection" in exc_str or "timeout" in exc_str:
                raise ProviderConnectionError(f"Network error: {exc}") from exc
            raise ProviderError(f"Ollama API error: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        choice = response.choices[0] if response.choices else None
        if choice is None:
            raise ResponseValidationError(f"Llama HF API returned no choices. Response: {response}")

        text = choice.message.content
        if not text or not text.strip():
            raise ResponseValidationError("Llama HF API returned empty content.")

        usage = getattr(response, "usage", None)

        return LLMResponse(
            text=text,
            finish_reason=str(choice.finish_reason) if choice.finish_reason else "unknown",
            latency_ms=latency_ms,
            tokens_input=getattr(usage, "prompt_tokens", 0) if usage else 0,
            tokens_output=getattr(usage, "completion_tokens", 0) if usage else 0,
            model_name=self._model,
            provider="llama",
            metadata={},
        )

    def shutdown(self) -> None:
        self._client = None
        logger.debug("Llama HF client shut down.")

    def provider_metadata(self) -> dict[str, Any]:
        return {
            "provider": "llama",
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            # Minimal inference call with a tiny prompt to verify connectivity
            self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                temperature=_MIN_TEMPERATURE,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Llama health check failed: %s", exc)
            return False


class LlamaAgent(Runtime):
    """Meta Llama agent for the LLM Reliability Ranking framework."""

    def __init__(self, config: Configuration) -> None:
        if config is None:
            raise ValueError("Configuration must be provided to LlamaAgent.")
        self._config = config
        self._adapter = _LlamaAdapter(config)
        self._max_retries: int = int(config.metadata.get("max_retries", 3))
        self._retry_backoff: float = float(config.metadata.get("retry_backoff", 1.0))
        self._rate_limiter = RateLimiter(
            requests_per_second=float(
                config.metadata.get("requests_per_second", DEFAULT_REQUESTS_PER_SECOND)
            )
        )

    def initialize(self) -> None:
        logger.info("Initialising LlamaAgent.")
        self._adapter.initialize()
        logger.info("LlamaAgent ready (model=%s).", self._adapter._model)

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
        logger.info(
            "LlamaAgent.run: task_id=%r, prompt_len=%d.",
            task.get("task_id", "<unknown>"),
            len(prompt),
        )
        self._rate_limiter.acquire()
        response = self._adapter.retry(
            request, max_attempts=self._max_retries, backoff_seconds=self._retry_backoff
        )
        logger.info(
            "LlamaAgent.run complete: task_id=%r, finish=%s.",
            task.get("task_id", "<unknown>"),
            response.finish_reason,
        )
        return response.text

    def shutdown(self) -> None:
        logger.info("Shutting down LlamaAgent.")
        self._adapter.shutdown()

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "LlamaAgent",
            "provider": "llama",
            "model": self._adapter._model,
            "version": LLAMA_AGENT_VERSION,
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


if not ProviderRegistry.exists("llama"):
    ProviderRegistry.register("llama", _LlamaAdapter)
if not RuntimeRegistry.exists("llama"):
    RuntimeRegistry.register("llama", LlamaAgent)
