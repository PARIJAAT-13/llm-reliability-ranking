"""Runtime adapters — fully capable inference backend implementations."""

from __future__ import annotations

from llm_reliability.runtime.adapters.llama_cpp import LlamaCppRuntime
from llm_reliability.runtime.adapters.lm_studio import LMStudioRuntime
from llm_reliability.runtime.adapters.mlx import MLXRuntime
from llm_reliability.runtime.adapters.ollama import OllamaRuntime
from llm_reliability.runtime.adapters.openai_compat import OpenAICompatRuntime
from llm_reliability.runtime.adapters.tgi import TGIRuntime
from llm_reliability.runtime.adapters.vllm import VLLMRuntime
from llm_reliability.runtime.registry import RuntimeRegistry


def _register_all() -> None:
    runtimes = {
        "ollama": OllamaRuntime,
        "vllm": VLLMRuntime,
        "llama.cpp": LlamaCppRuntime,
        "tgi": TGIRuntime,
        "lm-studio": LMStudioRuntime,
        "mlx": MLXRuntime,
        "openai-compat": OpenAICompatRuntime,
    }
    for name, cls in runtimes.items():
        if not RuntimeRegistry.exists(name):
            RuntimeRegistry.register(name, cls)


_register_all()

__all__ = [
    "OllamaRuntime",
    "VLLMRuntime",
    "LlamaCppRuntime",
    "TGIRuntime",
    "LMStudioRuntime",
    "MLXRuntime",
    "OpenAICompatRuntime",
]
