"""
Agent Factory for LLM Reliability Ranking Framework.

Resolves agent names to concrete Agent implementations.

The factory delegates to ``RuntimeRegistry`` which supports plugin-based
registration and discovery.  Third-party runtimes can be added without
modifying this file.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llm_reliability.runtime import Runtime, RuntimeRegistry

if TYPE_CHECKING:
    from llm_reliability.configs.config import Configuration

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prefix → canonical name map (backward-compatible name resolution)
# ---------------------------------------------------------------------------

_PREFIX_MAP: dict[str, str] = {
    "openai": "gpt",
    "gpt": "gpt",
    "gptAgent": "gpt",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "google": "gemini",
    "gemini": "gemini",
    "deepseek": "deepseek",
    "qwen": "qwen",
    "llama": "llama",
    "meta": "llama",
    "ollama": "ollama",
    "llamacpp": "llamacpp",
    "llama.cpp": "llamacpp",
    "vllm": "vllm",
    "huggingface": "huggingface",
    "hf": "huggingface",
    "mock": "mock",
    "mock_agent": "mock",
    "openrouter": "openrouter",
    "together": "together",
    "groq": "groq",
    "fireworks": "fireworks",
    "cohere": "cohere",
    "mistral": "mistral",
    "xai": "xai",
    "grok": "xai",
    "perplexity": "perplexity",
    "sonar": "perplexity",
    "azure": "azure_openai",
    "azure_openai": "azure_openai",
    "bedrock": "bedrock",
    "aws": "bedrock",
    "vertex": "vertex",
    "vertexai": "vertex",
    "sambanova": "sambanova",
    "cerebras": "cerebras",
    "nim": "nim",
    "nvidia": "nim",
    "litellm": "litellm",
    "sglang": "sglang",
}


def _resolve(name: str) -> str | None:
    """Resolve an agent name to a canonical runtime name, or ``None``.

    Checks ``RuntimeRegistry`` for an exact match first, then tries
    prefix-based resolution via ``_PREFIX_MAP``.
    """
    lower = name.lower()

    if RuntimeRegistry.exists(lower):
        return lower

    if lower in _PREFIX_MAP:
        return _PREFIX_MAP[lower]

    if ":" in name:
        provider = name.split(":")[0].lower()
        if provider in _PREFIX_MAP:
            return _PREFIX_MAP[provider]

    for prefix, canonical in _PREFIX_MAP.items():
        if lower.startswith(prefix):
            return canonical

    return None


class AgentFactory:
    """Create Agent instances from a name string and a Configuration."""

    @staticmethod
    def create(name: str, config: Configuration) -> Runtime:
        canonical = _resolve(name)
        if canonical is None:
            raise ValueError(
                f"Unknown agent name '{name}'. "
                f"Recognised prefixes: {sorted(_PREFIX_MAP.keys())}. "
                f"Use 'mock' or 'mock_agent' for testing."
            )

        try:
            runtime_cls = RuntimeRegistry.get(canonical)
        except ValueError:
            raise ValueError(
                f"Agent '{name}' resolved to canonical name '{canonical}' "
                f"but no runtime is registered under that name."
            )

        logger.debug("AgentFactory: resolved '%s' → %s", name, canonical)
        return runtime_cls(config=config)

    @staticmethod
    def is_mock(name: str) -> bool:
        """Return True if *name* resolves to MockAgent."""
        canonical = _resolve(name)
        return canonical == "mock"

    @staticmethod
    def available_names() -> list[str]:
        """Return the list of known agent name prefixes."""
        return sorted(_PREFIX_MAP.keys())

    @staticmethod
    def resolve(name: str) -> str | None:
        """Resolve *name* to a canonical runtime name, or ``None`` if unknown."""
        return _resolve(name)
