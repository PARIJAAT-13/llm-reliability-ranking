"""
VLLMRuntime — High-throughput vLLM inference backend.

Connects to a vLLM OpenAI-compatible REST server.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_reliability.runtime import Runtime
from llm_reliability.runtime.metadata import RuntimeMetadata

logger = logging.getLogger(__name__)


class VLLMRuntime(Runtime):
    """Full-capability vLLM runtime adapter."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "vllm-default",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._client: Any = None

    def initialize(self) -> None:
        try:
            from openai import OpenAI

            self._client = OpenAI(base_url=self._base_url, api_key="vllm")
        except ImportError as exc:
            raise ImportError("The 'openai' package is required for VLLMRuntime.") from exc
        logger.info("VLLMRuntime initialized (model=%s, url=%s).", self._model, self._base_url)

    def reset(self) -> None:
        pass

    def run(self, task: dict[str, Any]) -> Any:
        if self._client is None:
            raise RuntimeError("VLLMRuntime not initialized.")
        prompt = _extract_prompt(task)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def metadata(self) -> dict:
        return {"runtime": "vllm", "model": self._model, "base_url": self._base_url}

    def shutdown(self) -> None:
        self._client = None

    def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

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

    def runtime_metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(
            runtime_name="vllm",
            runtime_version=None,
            backend="vLLM",
            api_version="v1",
            execution_mode="server",
            gpu_acceleration=True,
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
