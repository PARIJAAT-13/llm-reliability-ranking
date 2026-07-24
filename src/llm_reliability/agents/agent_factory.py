"""
Agent Factory for LLM Reliability Ranking Framework.

Resolves agent names to concrete Agent implementations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llm_reliability.interfaces.agent import Agent

if TYPE_CHECKING:
    from llm_reliability.configs.config import Configuration

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry: maps lower-cased name prefix → import path + class name
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple[str, str]] = {
    # prefix                module path                                class name
    "openai":        ("llm_reliability.agents.gpt_agent",             "GPTAgent"),
    "gpt":           ("llm_reliability.agents.gpt_agent",             "GPTAgent"),
    "gptAgent":      ("llm_reliability.agents.gpt_agent",             "GPTAgent"),
    "anthropic":     ("llm_reliability.agents.anthropic_agent",       "AnthropicAgent"),
    "claude":        ("llm_reliability.agents.anthropic_agent",       "AnthropicAgent"),
    "google":        ("llm_reliability.agents.gemini_agent",          "GeminiAgent"),
    "gemini":        ("llm_reliability.agents.gemini_agent",          "GeminiAgent"),
    "deepseek":      ("llm_reliability.agents.deepseek_agent",        "DeepSeekAgent"),
    "qwen":          ("llm_reliability.agents.qwen_agent",            "QwenAgent"),
    "llama":         ("llm_reliability.agents.llama_agent",           "LlamaAgent"),
    "meta":          ("llm_reliability.agents.llama_agent",           "LlamaAgent"),
    "ollama":        ("llm_reliability.agents.ollama_agent",          "OllamaAgent"),
    "llamacpp":      ("llm_reliability.agents.llama_cpp_agent",       "LlamaCppAgent"),
    "llama.cpp":     ("llm_reliability.agents.llama_cpp_agent",       "LlamaCppAgent"),
    "vllm":          ("llm_reliability.agents.vllm_agent",            "VLLMAgent"),
    "huggingface":   ("llm_reliability.agents.hf_agent",              "HuggingFaceTransformersAgent"),
    "hf":            ("llm_reliability.agents.hf_agent",              "HuggingFaceTransformersAgent"),
    "mock":          ("llm_reliability.agents.mock_agent",            "MockAgent"),
    "mock_agent":    ("llm_reliability.agents.mock_agent",            "MockAgent"),
}


def _load_agent_class(module_path: str, class_name: str) -> type[Agent]:
    """Dynamically import and return an agent class."""
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls  # type: ignore[return-value]


def _resolve(name: str) -> tuple[str, str] | None:
    """Resolve an agent name to (module_path, class_name) or None."""
    lower = name.lower()

    if lower in _REGISTRY:
        return _REGISTRY[lower]

    if ":" in name:
        provider = name.split(":")[0].lower()
        if provider in _REGISTRY:
            return _REGISTRY[provider]

    for prefix, entry in _REGISTRY.items():
        if lower.startswith(prefix):
            return entry

    return None


class AgentFactory:
    """Create Agent instances from a name string and a Configuration."""

    @staticmethod
    def create(name: str, config: "Configuration") -> Agent:
        entry = _resolve(name)
        if entry is None:
            raise ValueError(
                f"Unknown agent name '{name}'. "
                f"Recognised prefixes: {sorted(_REGISTRY.keys())}. "
                f"Use 'mock' or 'mock_agent' for testing."
            )

        module_path, class_name = entry
        try:
            cls = _load_agent_class(module_path, class_name)
        except ImportError as exc:
            raise ImportError(
                f"Cannot load agent '{class_name}' from '{module_path}'. "
                f"Original error: {exc}"
            ) from exc

        logger.debug("AgentFactory: resolved '%s' → %s.%s", name, module_path, class_name)
        return cls(config=config)

    @staticmethod
    def is_mock(name: str) -> bool:
        """Return True if *name* resolves to MockAgent."""
        lower = name.lower()
        return lower in ("mock", "mock_agent")

    @staticmethod
    def available_names() -> list[str]:
        """Return the list of known agent name prefixes."""
        return sorted(_REGISTRY.keys())
