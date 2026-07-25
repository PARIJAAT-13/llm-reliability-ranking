"""
OpenAICompatRuntime — Generic OpenAI-compatible API runtime adapter.

Connects to any OpenAI-compatible API endpoint (OpenAI, Together AI,
Fireworks AI, Perplexity, Groq, etc.).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from llm_reliability.runtime import Runtime
from llm_reliability.runtime.metadata import RuntimeMetadata

logger = logging.getLogger(__name__)


class OpenAICompatRuntime(Runtime):
    """Generic OpenAI-compatible API runtime adapter."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "gpt-4o",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._timeout = timeout
        self._client: Any = None

    def initialize(self) -> None:
        try:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"timeout": self._timeout}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = OpenAI(**kwargs)
        except ImportError as exc:
            raise ImportError("The 'openai' package is required for OpenAICompatRuntime.") from exc
        logger.info("OpenAICompatRuntime initialized (model=%s).", self._model)

    def reset(self) -> None:
        pass

    def run(self, task: dict[str, Any]) -> Any:
        if self._client is None:
            raise RuntimeError("OpenAICompatRuntime not initialized.")
        prompt = _extract_prompt(task)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def metadata(self) -> dict:
        return {"runtime": "openai-compat", "model": self._model, "base_url": self._base_url}

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
            runtime_name="openai-compat",
            runtime_version=None,
            backend="openai-compatible",
            api_version="v1",
            execution_mode="api",
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
