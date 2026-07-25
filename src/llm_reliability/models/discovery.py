from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from llm_reliability.models.model_info import ModelInfo
from llm_reliability.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST: str = "http://127.0.0.1:11434"

FAMILY_MAP: dict[str, str] = {
    "llama": "Llama",
    "llama3": "Llama",
    "llama3.1": "Llama",
    "llama3.2": "Llama",
    "llama2": "Llama",
    "codellama": "Llama",
    "tinyllama": "TinyLlama",
    "qwen": "Qwen",
    "qwen2": "Qwen",
    "qwen2.5": "Qwen",
    "qwen2.5-coder": "Qwen",
    "qwen3": "Qwen",
    "gemma": "Gemma",
    "gemma2": "Gemma",
    "gemma3": "Gemma",
    "codegemma": "Gemma",
    "phi": "Phi",
    "phi3": "Phi",
    "phi3.5": "Phi",
    "phi4": "Phi",
    "phi4-mini": "Phi",
    "mistral": "Mistral",
    "mixtral": "Mistral",
    "mistral-nemo": "Mistral",
    "deepseek": "DeepSeek",
    "deepseek-r1": "DeepSeek",
    "deepseek-coder": "DeepSeek",
    "deepseek-coder-v2": "DeepSeek",
    "deepseek-v3": "DeepSeek",
    "smollm": "SmolLM",
    "smollm2": "SmolLM",
    "falcon": "Falcon",
    "falcon2": "Falcon",
    "falcon3": "Falcon",
    "yi": "Yi",
    "yi-coder": "Yi",
    "openchat": "OpenChat",
    "neural-chat": "NeuralChat",
    "starcoder": "StarCoder2",
    "starcoder2": "StarCoder2",
    "dbrx": "DBRX",
    "command-r": "Command R",
}


def _infer_family(identifier: str) -> str:
    base = identifier.split(":")[0].lower()
    for prefix, family in sorted(FAMILY_MAP.items(), key=lambda x: -len(x[0])):
        if base == prefix or base.startswith(prefix + "-") or base.startswith(prefix + "_"):
            return family
    return base.capitalize()


def _infer_size_gb(ollama_size_bytes: int) -> float | None:
    if ollama_size_bytes > 0:
        return round(ollama_size_bytes / (1024**3), 2)
    return None


def query_local_ollama_tags(
    base_url: str = DEFAULT_OLLAMA_HOST, timeout: float = 5.0
) -> list[dict[str, Any]]:
    tags_url = f"{base_url}/api/tags"
    req = urllib.request.Request(tags_url, headers={"User-Agent": "llm-reliability-ranking"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("models", [])


def discover_local_models(
    base_url: str = DEFAULT_OLLAMA_HOST, timeout: float = 5.0
) -> list[ModelInfo]:
    tags = query_local_ollama_tags(base_url, timeout)
    discovered: list[ModelInfo] = []
    for tag in tags:
        name = tag.get("name", "")
        if not name:
            continue
        if ModelRegistry.exists(name):
            continue
        size_bytes = tag.get("size", 0) or 0
        size_gb = _infer_size_gb(int(size_bytes))
        parameter_tag = tag.get("details", {}).get("parameter_size", "")
        if not parameter_tag:
            param_tag = name.split(":")[1] if ":" in name else "unknown"
            parameter_tag = param_tag.upper()
        param_count_str = parameter_tag.replace("B", "").strip()
        try:
            param_count = (
                float(param_count_str.split("x")[-1])
                if "x" not in param_count_str
                else float(param_count_str.split("x")[-1])
            )
        except ValueError:
            param_count = 0.0
        family = _infer_family(name)
        model_info = ModelInfo(
            family=family,
            name=f"{family} {parameter_tag} (local)",
            parameters=parameter_tag,
            parameter_count=param_count,
            context_window=0,
            recommended_ram_gb=size_gb * 1.5 if size_gb else None,
            recommended_vram_gb=size_gb,
            ollama_identifier=name,
            provider="ollama",
            runtime="ollama",
            status="supported",
            metadata={"discovered": True, "size_bytes": size_bytes, "family": family},
        )
        discovered.append(model_info)
    return discovered


def merge_discovered(
    discovered: list[ModelInfo], registry: type[ModelRegistry] = ModelRegistry
) -> int:
    count = 0
    for model in discovered:
        if not registry.exists(model.ollama_identifier):
            try:
                registry.register(model)
                count += 1
            except Exception as exc:
                logger.warning("Skipping model %s: %s", model.ollama_identifier, exc)
    return count
