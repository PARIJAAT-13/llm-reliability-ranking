from __future__ import annotations

import logging
import os
import time
from typing import Any

from llm_reliability.agents.adapters.exceptions import (AuthenticationError,
                                                        ProviderError)
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse
from llm_reliability.agents.utils.rate_limiter import RateLimiter
from llm_reliability.configs.config import Configuration
from llm_reliability.runtime.batching import BatchProcessor
from llm_reliability.runtime.cost_accounting import CostTracker
from llm_reliability.runtime.interface import Runtime
from llm_reliability.runtime.metadata import (RuntimeCapabilities,
                                              RuntimeMetadata)
from llm_reliability.runtime.streaming import TokenStream

logger = logging.getLogger(__name__)

_PROMPT_KEYS: tuple[str, ...] = ("prompt", "question", "problem_statement")


class BaseProvider(Runtime):
    """Abstract base class for all LLM providers, integrating Phase 1 runtime infrastructure.

    Subclasses can optionally override the template methods:
    ``_create_client()``, ``_call_api()``, ``_parse_response()``, ``_map_provider_error()``.

    Subclasses receive for free:
    - Configuration parsing (model, temperature, max_tokens, top_p, seed)
    - Rate limiting via ``RateLimiter``
    - Cost tracking via ``CostTracker``
    - Retry via ``retry()`` with configurable attempts/delay
    - Streaming via ``stream()`` returning ``TokenStream``
    - Batching via ``batch()`` returning ``list[LLMResponse]``
    - Failover across multiple providers via ``_resolve_failover()``
    - Structured error mapping via ``_call_api_safe()``
    - Standard ``metadata()``, ``runtime_metadata()``, ``_detect_capabilities()``
    - ``_extract_prompt()`` (shared, no more duplication)
    """

    provider_name: str = ""
    default_model: str = ""
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    default_top_p: float = 1.0
    default_requests_per_second: float = 10.0
    api_key_env: str = ""
    api_base_env: str = ""

    def __init__(self, config: Configuration) -> None:
        if config is None:
            raise ValueError("Configuration must be provided.")
        self._config = config
        self._client: Any = None
        _meta: dict[str, Any] = getattr(config, "metadata", {})
        self._model = _meta.get("model", getattr(config, "llm", "")) or self.default_model
        self._temperature = float(_meta.get("temperature", self.default_temperature))
        self._max_tokens = int(_meta.get("max_tokens", self.default_max_tokens))
        self._top_p = float(_meta.get("top_p", self.default_top_p))
        self._system_prompt: str | None = _meta.get("system_prompt")
        self._max_retries: int = int(_meta.get("max_retries", 3))
        self._retry_backoff: float = float(_meta.get("retry_backoff", 1.0))
        self._requests_per_second = float(
            _meta.get("requests_per_second", self.default_requests_per_second)
        )
        self._rate_limiter = RateLimiter(requests_per_second=self._requests_per_second)
        self._cost_tracker = CostTracker()
        self._batch_processor: BatchProcessor | None = None
        self._version: str = "1.0"
        _seed = getattr(config, "seed", 0)
        logger.debug(
            "%s created (model=%s, seed=%d, retries=%d).",
            self.__class__.__name__,
            self._model,
            _seed,
            self._max_retries,
        )

    # ------------------------------------------------------------------
    # Template methods — subclasses MAY override these
    # ------------------------------------------------------------------

    def _create_client(self) -> Any:
        """Create and return the provider SDK client.

        Override this if the provider creates its own client directly
        (rather than delegating to an adapter).

        Raises ``AuthenticationError`` if credentials are missing.
        """
        return None

    def _call_api(self, kwargs: dict[str, Any]) -> Any:
        """Make the actual API call to the provider.

        Override this if the provider does not use the adapter pattern.

        Parameters
        ----------
        kwargs:
            Provider-specific keyword arguments built by ``_build_request_kwargs()``.

        Returns
        -------
        Any
            Raw provider response object.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _call_api() or use an adapter."
        )

    def _parse_response(self, response: Any, latency_ms: float) -> LLMResponse:
        """Convert the raw provider response into a standard ``LLMResponse``.

        Override this if the provider does not use the adapter pattern.

        Parameters
        ----------
        response:
            Raw response object returned by ``_call_api()``.
        latency_ms:
            Measured wall-clock latency in milliseconds.

        Returns
        -------
        LLMResponse
            Standardised, provider-agnostic response.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _parse_response() or use an adapter."
        )

    def _map_provider_error(self, exc: Exception) -> ProviderError:
        """Map a provider-specific SDK exception to the framework's ``ProviderError`` hierarchy.

        Override this if the provider does not use the adapter pattern.

        Parameters
        ----------
        exc:
            The exception raised by the provider SDK.

        Returns
        -------
        ProviderError
            A typed framework exception with ``is_transient`` set appropriately.
        """
        if isinstance(exc, ProviderError):
            return exc
        return ProviderError(str(exc))

    # ------------------------------------------------------------------
    # Template methods — subclasses MAY override these
    # ------------------------------------------------------------------

    def _build_request_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        """Build the provider-specific API kwargs from an ``LLMRequest``.

        Override this method if the provider requires a different API shape.
        """
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        elif self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences
        return kwargs

    def _health_check_impl(self) -> bool:
        """Provider-specific health check.

        Default returns ``True`` (assumes healthy).
        """
        return True

    def _provider_metadata(self) -> dict[str, Any]:
        """Return provider-specific metadata.

        Default returns basic provider/model info.
        """
        return {
            "provider": self.provider_name,
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": self._top_p,
        }

    def _shutdown_client(self) -> None:
        """Release provider-specific resources.

        Override if the SDK client needs explicit cleanup.
        """

    # ------------------------------------------------------------------
    # Runtime / Agent interface implementation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Authenticate and create the provider SDK client.

        Reads the API key from the environment variable specified by ``api_key_env``.
        Delegates deferred SDK import and client construction to ``_create_client()``.

        Raises
        ------
        AuthenticationError
            If the API key environment variable is not set.
        ImportError
            If the provider SDK package is not installed.
        """
        logger.info("Initialising %s.", self.__class__.__name__)

        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise AuthenticationError(
                    f"{self.api_key_env} environment variable is not set. "
                    f"Set it before calling initialize()."
                )

        self._client = self._create_client()
        logger.info(
            "%s ready (model=%s).",
            self.__class__.__name__,
            self._model,
        )

    def reset(self) -> None:
        """Reset per-task state.

        Clears the cost tracker for a clean per-run audit trail.
        """
        self._cost_tracker = CostTracker()
        logger.debug("%s state reset.", self.__class__.__name__)

    def run(self, task: dict[str, Any]) -> Any:
        """Execute the provider on a benchmark task.

        Extracts the prompt, builds an ``LLMRequest``, applies rate limiting,
        calls the API with retry logic, tracks cost, and returns the raw text.

        Parameters
        ----------
        task:
            Task payload dict containing at least one of: ``prompt``, ``question``,
            ``problem_statement``.

        Returns
        -------
        str
            Raw text output from the model.
        """
        prompt = self._extract_prompt(task)
        request = self._build_request(prompt)

        logger.info(
            "%s.run: task_id=%r, model=%s, prompt_len=%d.",
            self.__class__.__name__,
            task.get("task_id", "<unknown>"),
            self._model,
            len(prompt),
        )

        self._rate_limiter.acquire()
        response = self._call_with_retry(request)
        self._track_cost(response)

        logger.info(
            "%s.run complete: task_id=%r, finish=%s, latency=%.1fms, "
            "tokens_in=%d, tokens_out=%d.",
            self.__class__.__name__,
            task.get("task_id", "<unknown>"),
            response.finish_reason,
            response.latency_ms,
            response.tokens_input,
            response.tokens_output,
        )

        return response.text

    def execute(self, task: dict[str, Any]) -> Any:
        """Execute a task and return the raw output.

        Delegates to ``run()`` for backward compatibility.
        """
        return self.run(task)

    def shutdown(self) -> None:
        """Release provider resources."""
        logger.info("Shutting down %s.", self.__class__.__name__)
        self._shutdown_client()
        self._client = None

    def metadata(self) -> dict[str, Any]:
        """Return descriptive metadata for logging and reproducibility."""
        return {
            "name": self.__class__.__name__,
            "provider": self.provider_name,
            "model": self._model,
            "version": self._version,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": self._top_p,
            "seed": getattr(self._config, "seed", 0),
            "max_retries": self._max_retries,
        }

    def runtime_metadata(self) -> RuntimeMetadata:
        """Return standardised metadata about this runtime instance."""
        base = super().runtime_metadata()
        return base.model_copy(
            update={
                "runtime_name": self.__class__.__name__,
                "backend": self.provider_name,
                "inference_parameters": {
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                    "top_p": self._top_p,
                    "model": self._model,
                },
                "capabilities": self._detect_capabilities(),
            }
        )

    def _detect_capabilities(self) -> RuntimeCapabilities:
        """Detect which optional capabilities this provider supports.

        Overridden methods (not using default from Runtime) are reported as
        supported.  Batch and streaming support are detected from the class.
        """
        caps = super()._detect_capabilities()
        return caps.model_copy(
            update={
                "batch_inference": hasattr(self, "batch"),
                "streaming": hasattr(self, "stream"),
                "gpu_acceleration": self._detect_gpu(),
            }
        )

    def _detect_gpu(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def health_check(self) -> bool:
        """Return True if the provider is reachable and responsive."""
        if self._client is None:
            return False
        try:
            return self._health_check_impl()
        except Exception as exc:
            logger.warning("%s health check failed: %s", self.__class__.__name__, exc)
            return False

    # ------------------------------------------------------------------
    # Retry / failover
    # ------------------------------------------------------------------

    def retry(
        self,
        request: LLMRequest,
        max_attempts: int | None = None,
        backoff_seconds: float | None = None,
    ) -> LLMResponse:
        """Call the API with exponential-backoff retry.

        Retries only on transient errors (rate limits, network issues, timeouts).
        Fails fast on non-transient errors (auth, validation, not-found, quota).
        """
        max_attempts = max_attempts or self._max_retries
        backoff_seconds = backoff_seconds or self._retry_backoff
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return self._call_with_latency(request)
            except ProviderError as exc:
                last_exc = exc
                if not exc.is_transient or attempt == max_attempts:
                    raise
                wait = backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Transient error on attempt %d/%d: %s — retrying in %.1fs",
                    attempt,
                    max_attempts,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        request: LLMRequest,
        timeout: float | None = None,
    ) -> TokenStream:
        """Stream tokens from the provider.

        Parameters
        ----------
        request:
            The ``LLMRequest`` to send.
        timeout:
            Optional timeout in seconds for the entire stream.

        Returns
        -------
        TokenStream
            An iterable stream of tokens.
        """
        kwargs = self._build_request_kwargs(request)
        kwargs["stream"] = True

        def token_generator() -> Any:
            response = self._call_api_safe(kwargs)
            yield from self._extract_stream_tokens(response)

        return TokenStream(generator=token_generator(), timeout=timeout)

    def _extract_stream_tokens(self, response: Any) -> Any:
        """Extract tokens from a streaming response.

        Override this if the provider uses a non-standard streaming format.
        Default implementation assumes standard OpenAI-style chunk iteration.
        """
        for chunk in response:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def batch(
        self,
        tasks: list[dict[str, Any]],
        max_batch_size: int = 16,
    ) -> list[LLMResponse]:
        """Process multiple tasks in batches.

        Parameters
        ----------
        tasks:
            List of task dicts.
        max_batch_size:
            Maximum number of tasks per batch.

        Returns
        -------
        list[LLMResponse]
            Responses for each task.
        """
        processor = BatchProcessor(
            executor=self._make_batch_executor(),
            max_batch_size=max_batch_size,
        )
        results = processor.process_all(tasks)
        responses: list[LLMResponse] = []
        for batch_result in results:
            for r in batch_result.results:
                if isinstance(r, LLMResponse):
                    responses.append(r)
        return responses

    def _make_batch_executor(self) -> Any:
        class _BatchWrapper:
            def __init__(self, outer: BaseProvider) -> None:
                self._outer = outer

            def execute(self, task: dict[str, Any]) -> Any:
                prompt = self._outer._extract_prompt(task)
                request = self._outer._build_request(prompt)
                return self._outer._call_with_latency(request)

        return _BatchWrapper(self)

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    def _track_cost(self, response: LLMResponse) -> None:
        self._cost_tracker.record_call(
            provider=self.provider_name,
            model=self._model,
            input_tokens=response.tokens_input,
            output_tokens=response.tokens_output,
            latency_ms=response.latency_ms,
        )

    def cost_summary(self) -> dict[str, Any]:
        return self._cost_tracker.summary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_request(self, prompt: str) -> LLMRequest:
        _seed = getattr(self._config, "seed", None)
        return LLMRequest(
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            top_p=self._top_p,
            seed=_seed if _seed is not None else None,
            system_prompt=self._system_prompt,
        )

    def _call_with_latency(self, request: LLMRequest) -> LLMResponse:
        kwargs = self._build_request_kwargs(request)
        t0 = time.perf_counter()
        response = self._call_api_safe(kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return self._parse_response(response, latency_ms)

    def _call_api_safe(self, kwargs: dict[str, Any]) -> Any:
        try:
            return self._call_api(kwargs)
        except Exception as exc:
            raise self._map_provider_error(exc) from exc

    def _call_with_retry(self, request: LLMRequest) -> LLMResponse:
        return self.retry(request)

    @staticmethod
    def _extract_prompt(task: dict[str, Any]) -> str:
        for key in _PROMPT_KEYS:
            value = task.get(key)
            if value and str(value).strip():
                return str(value).strip()
        fallback = str(task).strip()
        if not fallback or fallback == "{}":
            raise ValueError(
                f"Cannot extract a prompt from task dict: {task!r}. "
                f"Task must contain one of: {_PROMPT_KEYS}."
            )
        logger.warning(
            "No standard prompt key found in task %r; using str(task) as prompt.",
            task.get("task_id", "<unknown>"),
        )
        return fallback
