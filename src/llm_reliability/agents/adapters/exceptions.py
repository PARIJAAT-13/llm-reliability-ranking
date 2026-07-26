"""
Custom exceptions for the LLM Agent Adapter Framework.

These are raised at different stages of the request/response lifecycle
so callers can handle each failure mode explicitly.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for all provider-related errors."""

    is_transient: bool = True


class RateLimitError(ProviderError):
    """Raised when the provider returns a rate-limit response."""

    is_transient: bool = True


class AuthenticationError(ProviderError):
    """Raised when the provider rejects credentials."""

    is_transient: bool = False


class RequestValidationError(ProviderError):
    """Raised when an LLMRequest fails structural validation."""

    is_transient: bool = False


class ResponseValidationError(ProviderError):
    """Raised when an LLMResponse from a provider is structurally invalid."""

    is_transient: bool = False


class ConnectionError(ProviderError):
    """Raised when network connectivity to the provider fails."""

    is_transient: bool = True


class OllamaServerNotFoundError(ConnectionError):
    """Raised when the local Ollama server is not running or unreachable."""

    is_transient: bool = False


class OllamaModelNotFoundError(ProviderError):
    """Raised when a requested model is not installed on the local Ollama server."""

    is_transient: bool = False


class OllamaMemoryError(ProviderError):
    """Raised when a model cannot be loaded due to insufficient system RAM or GPU VRAM."""

    is_transient: bool = False


class TimeoutError(ProviderError):
    """Raised when a provider request exceeds the configured timeout."""

    is_transient: bool = True


class QuotaExceededError(ProviderError):
    """Raised when the provider account has exceeded its usage quota."""

    is_transient: bool = False


class ProviderUnavailableError(ProviderError):
    """Raised when the provider service is unavailable (e.g., maintenance, overloaded)."""

    is_transient: bool = True


class InvalidRequestError(ProviderError):
    """Raised when the provider rejects the request as invalid (bad parameters, unsupported features)."""

    is_transient: bool = False


class NetworkError(ProviderError):
    """Raised when a network-level failure occurs (DNS, connection reset, SSL issues)."""

    is_transient: bool = True


class ContentFilterError(ProviderError):
    """Raised when the provider's content filter blocks the response."""

    is_transient: bool = False


class ContextLengthExceededError(InvalidRequestError):
    """Raised when the input exceeds the provider's maximum context window."""

    is_transient: bool = False
