"""
Abstract base class for all LLM provider adapters.

BaseLLMAdapter provides shared infrastructure — configuration, logging,
request/response validation, latency measurement, and retry logic — so
concrete provider adapters only need to implement API-specific behaviour.

How to create a new provider adapter
-------------------------------------
1. Inherit from BaseLLMAdapter.
2. Implement the five abstract methods:
   - initialize()    – authenticate / warm up connections
   - generate()      – call the provider API and return LLMResponse
   - shutdown()      – release resources
   - provider_metadata() – return a dict of provider-specific info
   - health_check()  – verify the provider is reachable
3. Register the adapter with ProviderRegistry:
   ProviderRegistry.register("MyProvider", MyProviderAdapter)
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from llm_reliability.agents.adapters.exceptions import (
    ProviderError,
    RequestValidationError,
    ResponseValidationError,
)
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse
from llm_reliability.configs.config import Configuration

logger = logging.getLogger(__name__)


class BaseLLMAdapter(ABC):
    """Abstract base class for all LLM provider adapters."""

    def __init__(self, config: Configuration) -> None:
        """Initialize the adapter with a framework configuration."""
        if config is None:
            raise ValueError("Configuration must be provided.")
        self.config = config
        self._request_logs: list[dict[str, Any]] = []
        self._response_logs: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by every provider adapter
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """Authenticate and warm up provider connections."""

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Call the provider API and return a validated LLMResponse."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release any resources held by this adapter."""

    @abstractmethod
    def provider_metadata(self) -> dict[str, Any]:
        """Return provider-specific metadata (model versions, limits, etc.)."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the provider is reachable and responsive."""

    # ------------------------------------------------------------------
    # Concrete shared methods — used by all provider adapters
    # ------------------------------------------------------------------

    def validate_request(self, request: LLMRequest) -> None:
        """Validate a request before sending it to the provider.

        Raises RequestValidationError on invalid input.
        """
        if not isinstance(request, LLMRequest):
            raise RequestValidationError(
                f"Expected LLMRequest, got {type(request).__name__}."
            )

    def validate_response(self, response: LLMResponse) -> None:
        """Validate a response received from the provider.

        Raises ResponseValidationError if the response is structurally invalid.
        """
        if not isinstance(response, LLMResponse):
            raise ResponseValidationError(
                f"Expected LLMResponse, got {type(response).__name__}."
            )
        if not response.text.strip():
            raise ResponseValidationError("LLMResponse.text must not be blank.")

    def log_request(self, request: LLMRequest) -> None:
        """Log a request payload for audit and debugging."""
        entry = {
            "event": "request",
            "provider": self.__class__.__name__,
            "prompt_length": len(request.prompt),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "seed": request.seed,
        }
        self._request_logs.append(entry)
        logger.debug("LLM request: %s", entry)

    def log_response(self, response: LLMResponse) -> None:
        """Log a response for audit and debugging."""
        entry = {
            "event": "response",
            "provider": response.provider,
            "model": response.model_name,
            "finish_reason": response.finish_reason,
            "latency_ms": response.latency_ms,
            "tokens_input": response.tokens_input,
            "tokens_output": response.tokens_output,
        }
        self._response_logs.append(entry)
        logger.debug("LLM response: %s", entry)

    def measure_latency(self, request: LLMRequest) -> tuple[LLMResponse, float]:
        """Invoke generate() and return (response, latency_ms).

        Validates the request before and the response after each call.
        """
        self.validate_request(request)
        self.log_request(request)
        start = time.perf_counter()
        response = self.generate(request)
        latency_ms = (time.perf_counter() - start) * 1000.0
        self.validate_response(response)
        self.log_response(response)
        return response, latency_ms

    def retry(
        self,
        request: LLMRequest,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
    ) -> LLMResponse:
        """Invoke generate() with exponential back-off retry.

        Retries on any ProviderError subclass.
        Raises the final exception if all attempts are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response, _ = self.measure_latency(request)
                return response
            except ProviderError as exc:
                last_exc = exc
                wait = backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Provider error on attempt %d/%d: %s — retrying in %.1fs",
                    attempt, max_attempts, exc, wait,
                )
                if attempt < max_attempts:
                    time.sleep(wait)
        raise last_exc  # type: ignore[misc]
