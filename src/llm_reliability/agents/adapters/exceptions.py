"""
Custom exceptions for the LLM Agent Adapter Framework.

These are raised at different stages of the request/response lifecycle
so callers can handle each failure mode explicitly.
"""


class ProviderError(Exception):
    """Base class for all provider-related errors."""


class RateLimitError(ProviderError):
    """Raised when the provider returns a rate-limit response."""


class AuthenticationError(ProviderError):
    """Raised when the provider rejects credentials."""


class RequestValidationError(ProviderError):
    """Raised when an LLMRequest fails structural validation."""


class ResponseValidationError(ProviderError):
    """Raised when an LLMResponse from a provider is structurally invalid."""


class ConnectionError(ProviderError):
    """Raised when network connectivity to the provider fails."""
