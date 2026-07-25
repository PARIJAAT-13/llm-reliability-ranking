"""
OllamaRuntime — fully capable Ollama inference backend.

Extends the basic ``Runtime`` interface with health checking, token counting,
memory measurement, version detection, and runtime metadata.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

from llm_reliability.agents.utils.ollama_utils import (
    check_ollama_server,
    estimate_model_memory,
    get_available_memory_gb,
    list_local_models,
    normalize_ollama_url,
    unload_ollama_model,
)
from llm_reliability.runtime import Runtime
from llm_reliability.runtime.metadata import RuntimeMetadata

logger = logging.getLogger(__name__)


class OllamaRuntime(Runtime):
    """Full-capability Ollama runtime adapter.

    Supports: health checking, version detection, model loading/unloading,
    memory estimation, token counting.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "llama3.1:8b",
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._client: Any = None
        self._version: str | None = None
        self._available_models: list[str] = []

    def initialize(self) -> None:
        try:
            from openai import OpenAI

            api_url = normalize_ollama_url(self._base_url)
            self._client = OpenAI(base_url=api_url, api_key="ollama")
        except ImportError as exc:
            raise ImportError("The 'openai' package is required for OllamaRuntime.") from exc
        self._detect_version()
        self._refresh_available_models()
        logger.info(
            "OllamaRuntime initialized (model=%s, url=%s, version=%s).",
            self._model,
            self._base_url,
            self._version or "unknown",
        )

    def reset(self) -> None:
        pass

    def run(self, task: dict[str, Any]) -> Any:
        if self._client is None:
            raise RuntimeError("OllamaRuntime not initialized.")
        prompt = _extract_prompt(task)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def metadata(self) -> dict:
        return {"runtime": "ollama", "model": self._model, "base_url": self._base_url}

    def shutdown(self) -> None:
        unload_ollama_model(self._model, base_url=self._base_url)
        self._client = None

    def health_check(self) -> bool:
        result = check_ollama_server(base_url=self._base_url)
        if isinstance(result, tuple):
            return bool(result[0])
        return bool(result)

    def load_model(self) -> None:
        """Ensure the model is loaded by sending a warm-up request."""
        if self._client is None:
            self.initialize()
        if not self.health_check():
            raise RuntimeError(f"Ollama server not reachable at {self._base_url}")
        try:
            self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "warmup"}],
                temperature=0.0,
                max_tokens=1,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load model {self._model}: {exc}") from exc

    def unload_model(self) -> None:
        unload_ollama_model(self._model, base_url=self._base_url)

    def count_tokens(self, text: str) -> int:
        if self._client is None:
            return 0
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": text}],
                temperature=0.0,
                max_tokens=1,
            )
            usage = getattr(resp, "usage", None)
            if usage:
                return usage.prompt_tokens
        except Exception:
            pass
        return 0

    def measure_memory(self) -> dict[str, float]:
        mem = {}
        try:
            avail = get_available_memory_gb()
            if avail:
                mem["available_ram_gb"] = avail
            est = estimate_model_memory(self._model, base_url=self._base_url)
            if est:
                mem["estimated_model_memory_gb"] = est
        except Exception:
            pass
        return mem

    def runtime_metadata(self) -> RuntimeMetadata:
        models = list_local_models(base_url=self._base_url) if self.health_check() else []
        mem_info = self.measure_memory()
        return RuntimeMetadata(
            runtime_name="ollama",
            runtime_version=self._version,
            backend="ollama",
            api_version="v1",
            execution_mode="local",
            gpu_acceleration="gpu" in str(mem_info.get("available_ram_gb", "")).lower(),
            metadata={
                "model": self._model,
                "base_url": self._base_url,
                "available_models": models,
                "memory": mem_info,
            },
        )

    def _detect_version(self) -> None:
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            output = result.stdout.strip() or result.stderr.strip()
            if output and "Warning" not in output:
                self._version = output
        except Exception:
            self._version = None
        if self._version is None and self._client is not None:
            try:
                resp = self._client.models.list()
                self._version = getattr(resp, "model", None) or None
            except Exception:
                pass

    def _refresh_available_models(self) -> None:
        try:
            self._available_models = list_local_models(base_url=self._base_url)
        except Exception:
            self._available_models = []


def _extract_prompt(task: dict[str, Any]) -> str:
    for key in ("prompt", "question", "problem_statement"):
        val = task.get(key)
        if val and isinstance(val, str):
            return val
    return str(task)
