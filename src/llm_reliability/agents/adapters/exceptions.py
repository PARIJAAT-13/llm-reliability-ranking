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
