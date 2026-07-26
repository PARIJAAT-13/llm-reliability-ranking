"""
LMStudioRuntime — LM Studio inference backend.

Connects to the LM Studio local inference server via its OpenAI-compatible API.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_reliability.runtime import Runtime
from llm_reliability.runtime.metadata import RuntimeMetadata

logger = logging.getLogger(__name__)


class LMStudioRuntime(Runtime):
    """LM Studio runtime adapter."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        model: str = "lm-studio-model",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._client: Any = None

    def initialize(self) -> None:
        try:
            from openai import OpenAI

            self._client = OpenAI(base_url=self._base_url, api_key="lm-studio")
        except ImportError as exc:
            raise ImportError("The 'openai' package is required for LMStudioRuntime.") from exc
        logger.info(
            "LMStudioRuntime initialized (model=%s, url=%s).",
            self._model,
            self._base_url,
        )

    def reset(self) -> None:
        pass

    def run(self, task: dict[str, Any]) -> Any:
        if self._client is None:
            raise RuntimeError("LMStudioRuntime not initialized.")
        prompt = _extract_prompt(task)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def metadata(self) -> dict:
        return {
            "runtime": "lm-studio",
            "model": self._model,
            "base_url": self._base_url,
        }

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

    def runtime_metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(
            runtime_name="lm-studio",
            runtime_version=None,
            backend="lm-studio",
            api_version="v1",
            execution_mode="local",
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
