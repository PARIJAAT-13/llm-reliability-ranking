"""
Agent Factory for LLM Reliability Ranking Framework.

Resolves agent names to concrete Agent implementations.

Name Resolution Order
---------------------
1.  Exact class name (e.g. ``"GPTAgent"``)
2.  Provider shorthand registered in the AGENT_REGISTRY
    (e.g. ``"openai"``, ``"gpt-4o"``, ``"gpt-4.1"``)
3.  Fallback to MockAgent **only** when the name is ``"mock"`` or
    ``"mock_agent"``; any other unknown name raises ``ValueError``.

Supported Provider Names
------------------------
========================  ========================  =====================
Agent name(s)             Class                     Env var required
========================  ========================  =====================
``openai``, ``gpt-*``     GPTAgent                  OPENAI_API_KEY
``anthropic``, ``claude*``AnthropicAgent            ANTHROPIC_API_KEY
``google``, ``gemini*``   GeminiAgent               GEMINI_API_KEY
``deepseek``              DeepSeekAgent             DEEPSEEK_API_KEY
``qwen``                  QwenAgent                 QWEN_API_KEY
``llama``, ``meta*``      LlamaAgent                HF_TOKEN
``mock``, ``mock_agent``  MockAgent                 (none)
========================  ========================  =====================

Usage
-----
>>> from llm_reliability.agents.agent_factory import AgentFactory
>>> from llm_reliability.configs.config import Configuration
>>> cfg = Configuration(...)
>>> agent = AgentFactory.create("openai:gpt-4o", cfg)
>>> agent.initialize()
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
# We use deferred imports so that missing provider SDKs don't break startup.

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

    # 1. Exact lower-cased key match
    if lower in _REGISTRY:
        return _REGISTRY[lower]

    # 2. Provider:model syntax  e.g. "openai:gpt-4o"
    if ":" in name:
        provider = name.split(":")[0].lower()
        if provider in _REGISTRY:
            return _REGISTRY[provider]

    # 3. Prefix match (handles "gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro", …)
    for prefix, entry in _REGISTRY.items():
        if lower.startswith(prefix):
            return entry

    return None


class AgentFactory:
    """Create Agent instances from a name string and a Configuration."""

    @staticmethod
    def create(name: str, config: "Configuration") -> Agent:
        """Instantiate the appropriate Agent for *name*.

        Parameters
        ----------
        name:
            Agent identifier.  Can be a class name, a provider shorthand,
            a ``provider:model`` compound, or a model name prefix.
        config:
            Framework Configuration object passed to the agent constructor.

        Returns
        -------
        Agent
            An un-initialized agent instance (caller must call
            ``agent.initialize()``).

        Raises
        ------
        ValueError
            If *name* is not recognised and is not the mock sentinel.
        """
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
            # Give a helpful message if the optional SDK is not installed
            raise ImportError(
                f"Cannot load agent '{class_name}' from '{module_path}'. "
                f"The provider SDK may not be installed. "
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
