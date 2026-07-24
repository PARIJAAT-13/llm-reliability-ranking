"""
OllamaAgent — Generic Ollama provider adapter for LLM Reliability Ranking.

Connects to a local Ollama server via its OpenAI-compatible REST endpoint
(default: http://127.0.0.1:11434/v1). Supports any local Ollama model by
reading the model name from `config.metadata["model"]`.

Provides pre-execution validation, memory estimation, smart error handling,
client reuse, and automatic model unloading upon shutdown (`keep_alive=0`).

    OllamaAgent(Agent)
      └── _OllamaAdapter(BaseLLMAdapter)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from llm_reliability.agents.adapters.base_llm_adapter import BaseLLMAdapter
from llm_reliability.agents.adapters.exceptions import (
    AuthenticationError,
    ConnectionError as ProviderConnectionError,
    OllamaMemoryError,
    OllamaModelNotFoundError,
    OllamaServerNotFoundError,
    ProviderError,
    RateLimitError,
    ResponseValidationError,
)
from llm_reliability.agents.adapters.provider_registry import ProviderRegistry
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse
from llm_reliability.agents.utils.ollama_utils import (
    check_ollama_server,
    estimate_model_memory,
    format_memory_error,
    format_model_not_found_error,
    get_available_memory_gb,
    list_local_models,
    model_matches,
    normalize_ollama_url,
    unload_ollama_model,
)
from llm_reliability.agents.utils.rate_limiter import RateLimiter
from llm_reliability.configs.config import Configuration
from llm_reliability.runtime import Runtime
from llm_reliability.runtime.registry import RuntimeRegistry

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "llama3.1:8b"
DEFAULT_BASE_URL: str = "http://127.0.0.1:11434/v1"
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_REQUESTS_PER_SECOND: float = 10.0
OLLAMA_AGENT_VERSION: str = "1.1.0"

_PROMPT_KEYS: tuple[str, ...] = ("prompt", "question", "problem_statement")


class _OllamaAdapter(BaseLLMAdapter):
    """Internal Ollama adapter using a reusable OpenAI client pointed to Ollama."""

    _client: Any

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._client = None
        configured_model = config.metadata.get("model") or config.llm
        if not configured_model or configured_model in ("ollama", "mock"):
            configured_model = DEFAULT_MODEL
        self._model = configured_model

        self._base_url = config.metadata.get("base_url") or DEFAULT_BASE_URL
        self._temperature = float(config.metadata.get("temperature", DEFAULT_TEMPERATURE))
        self._max_tokens = int(config.metadata.get("max_tokens", DEFAULT_MAX_TOKENS))
        self._system_prompt: str | None = config.metadata.get("system_prompt")
        self._host_url = normalize_ollama_url(self._base_url)

    def initialize(self) -> None:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OllamaAgent. "
                "Install with: pip install openai"
            ) from exc

        # Task 3: Pre-validation server connectivity
        server_ok, msg = check_ollama_server(self._host_url)
        if not server_ok:
            err_msg = f"Ollama server not running or unreachable at {self._host_url}: {msg}"
            logger.error(err_msg)
            raise OllamaServerNotFoundError(err_msg)

        # Task 3: Validate model exists
        installed = list_local_models(self._host_url)
        if installed and not model_matches(self._model, installed):
            err_msg = format_model_not_found_error([self._model], installed)
            logger.error(err_msg)
            raise OllamaModelNotFoundError(err_msg)

        # Task 4: Check memory before loading
        mem_info = estimate_model_memory(self._model, self._host_url)
        avail_ram = get_available_memory_gb()
        model_size_gb = mem_info.get("size_gb")
        if model_size_gb and avail_ram and model_size_gb > avail_ram:
            err_msg = format_memory_error(self._model, model_size_gb, avail_ram)
            logger.warning("Pre-load memory check warning: %s", err_msg)

        # Task 13: Reuse HTTP client instance
        if self._client is None:
            self._client = OpenAI(
                base_url=f"{self._host_url}/v1",
                api_key="ollama",
            )

        logger.info("Initializing OllamaAgent\nModel: %s", self._model)
        logger.info("Provider : Ollama | Model : %s | URL : %s", self._model, self._base_url)

    def generate(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("_OllamaAdapter.generate() called before initialize().")

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
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            exc_str = str(exc).lower()
            # 1. Model not found (HTTP 404)
            if "not found" in exc_str or "404" in exc_str or ("model" in exc_str and "does not exist" in exc_str):
                installed = list_local_models(self._host_url)
                raise OllamaModelNotFoundError(format_model_not_found_error([self._model], installed)) from exc
            # 2. System memory / VRAM / CUDA allocation failure (HTTP 500 or memory error message)
            is_memory_err = (
                any(term in exc_str for term in ("memory", "ram", "vram", "cuda", "alloc", "system memory", "requires", "gib", "mib", "out of memory", "insufficient"))
                or ("500" in exc_str and any(t in exc_str for t in ("internal", "server", "error", "model", "load")))
            )
            if is_memory_err:
                avail_ram = get_available_memory_gb()
                raise OllamaMemoryError(format_memory_error(self._model, None, avail_ram)) from exc
            if any(term in exc_str for term in ("connection", "refused", "10061", "111", "closed", "reset")):
                raise OllamaServerNotFoundError(f"Ollama server connection lost: {exc}") from exc
            if "unauthorized" in exc_str or "401" in exc_str:
                raise AuthenticationError(f"Authentication failed: {exc}") from exc
            if "rate limit" in exc_str or "429" in exc_str or "too many" in exc_str:
                raise RateLimitError(f"Rate limit exceeded: {exc}") from exc
            if "timeout" in exc_str:
                raise ProviderConnectionError(f"Network timeout contacting Ollama: {exc}") from exc
            raise ProviderError(f"Ollama API error: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        choice = response.choices[0] if response.choices else None
        if choice is None:
            raise ResponseValidationError(f"Ollama API returned no choices. Response: {response}")

        text = choice.message.content
        if not text or not text.strip():
            raise ResponseValidationError("Ollama API returned empty content.")

        usage = getattr(response, "usage", None)

        return LLMResponse(
            text=text,
            finish_reason=str(choice.finish_reason) if choice.finish_reason else "unknown",
            latency_ms=latency_ms,
            tokens_input=getattr(usage, "prompt_tokens", 0) if usage else 0,
            tokens_output=getattr(usage, "completion_tokens", 0) if usage else 0,
            model_name=self._model,
            provider="ollama",
            metadata={},
        )

    def shutdown(self) -> None:
        # Task 6: Automatic Model Cleanup via keep_alive: 0
        if self._model:
            unload_ollama_model(self._model, self._host_url)
        self._client = None
        logger.debug("Ollama client shut down.")

    def provider_metadata(self) -> dict[str, Any]:
        return {
            "provider": "ollama",
            "model": self._model,
            "base_url": self._base_url,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    def health_check(self) -> bool:
        server_ok, _ = check_ollama_server(self._host_url, timeout=2.0)
        return server_ok


class OllamaAgent(Runtime):
    """Generic Ollama agent for the LLM Reliability Ranking framework."""

    def __init__(self, config: Configuration) -> None:
        if config is None:
            raise ValueError("Configuration must be provided to OllamaAgent.")
        self._config = config
        self._adapter = _OllamaAdapter(config)
        self._max_retries: int = int(config.metadata.get("max_retries", 3))
        self._retry_backoff: float = float(config.metadata.get("retry_backoff", 1.0))
        self._rate_limiter = RateLimiter(
            requests_per_second=float(
                config.metadata.get("requests_per_second", DEFAULT_REQUESTS_PER_SECOND)
            )
        )

    def initialize(self) -> None:
        logger.info("Initializing OllamaAgent")
        self._adapter.initialize()
        logger.info("OllamaAgent ready (model=%s).", self._adapter._model)

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
        logger.info("OllamaAgent.run: task_id=%r, prompt_len=%d.",
                    task.get("task_id", "<unknown>"), len(prompt))
        self._rate_limiter.acquire()
        response = self._adapter.retry(request, max_attempts=self._max_retries,
                                       backoff_seconds=self._retry_backoff)
        logger.info("OllamaAgent.run complete: task_id=%r, finish=%s.",
                    task.get("task_id", "<unknown>"), response.finish_reason)
        return response.text

    def shutdown(self) -> None:
        logger.info("Shutting down OllamaAgent.")
        self._adapter.shutdown()

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "OllamaAgent",
            "provider": "ollama",
            "model": self._adapter._model,
            "version": OLLAMA_AGENT_VERSION,
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


if not ProviderRegistry.exists("ollama"):
    ProviderRegistry.register("ollama", _OllamaAdapter)
if not RuntimeRegistry.exists("ollama"):
    RuntimeRegistry.register("ollama", OllamaAgent)
