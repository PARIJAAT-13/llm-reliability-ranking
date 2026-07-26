"""
LlamaCppRuntime — llama.cpp inference backend.

Connects to a local llama-server REST endpoint or falls back to
llama_cpp Python bindings.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_reliability.runtime import Runtime
from llm_reliability.runtime.metadata import RuntimeMetadata

logger = logging.getLogger(__name__)


class LlamaCppRuntime(Runtime):
    """Full-capability llama.cpp runtime adapter."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/completion",
        model: str = "llama.cpp-local",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._client: Any = None
        self._use_bindings = False

    def initialize(self) -> None:
        try:
            import httpx

            self._client = httpx.Client(timeout=self._timeout)
        except ImportError as exc:
            raise ImportError("The 'httpx' package is required for LlamaCppRuntime.") from exc
        logger.info(
            "LlamaCppRuntime initialized (model=%s, url=%s).",
            self._model,
            self._base_url,
        )

    def reset(self) -> None:
        pass

    def run(self, task: dict[str, Any]) -> Any:
        if self._client is None:
            raise RuntimeError("LlamaCppRuntime not initialized.")
        prompt = _extract_prompt(task)
        try:
            resp = self._client.post(
                self._base_url,
                json={
                    "prompt": prompt,
                    "temperature": 0.0,
                    "max_tokens": 1024,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("content", data.get("text", ""))
        except Exception as exc:
            raise RuntimeError(f"llama.cpp inference failed: {exc}") from exc

    def metadata(self) -> dict:
        return {
            "runtime": "llama.cpp",
            "model": self._model,
            "base_url": self._base_url,
        }

    def shutdown(self) -> None:
        self._client = None

    def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            resp = self._client.get(self._base_url.replace("/completion", "/health"))
            return resp.status_code == 200
        except Exception:
            return False

    def runtime_metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(
            runtime_name="llama.cpp",
            runtime_version=None,
            backend="llama.cpp",
            execution_mode="local",
            gpu_acceleration=False,
            metadata={
                "model": self._model,
                "base_url": self._base_url,
            },
        )


def _extract_prompt(task: dict[str, Any]) -> str:
    for key in ("prompt", "question", "problem_statement"):
        val = task.get(key)
        if val and isinstance(val, str):
            return val
    return str(task)
