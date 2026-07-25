"""
MLXRuntime — Apple MLX inference backend.

Connects to the MLX community server or uses mlx-lm Python package directly.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_reliability.runtime import Runtime
from llm_reliability.runtime.metadata import RuntimeMetadata

logger = logging.getLogger(__name__)


class MLXRuntime(Runtime):
    """Apple MLX runtime adapter."""

    def __init__(
        self,
        model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        max_tokens: int = 1024,
        temp: float = 0.0,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temp = temp
        self._loaded = False
        self._model_obj: Any = None
        self._tokenizer: Any = None

    def initialize(self) -> None:
        logger.info("MLXRuntime initialized (model=%s).", self._model)
        self._loaded = True

    def reset(self) -> None:
        pass

    def run(self, task: dict[str, Any]) -> Any:
        prompt = _extract_prompt(task)
        try:
            from mlx_lm import generate, load

            if not self._loaded or self._model_obj is None:
                self._model_obj, self._tokenizer = load(self._model)
                self._loaded = True
            response = generate(
                self._model_obj,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self._max_tokens,
                temp=self._temp,
            )
            return response
        except ImportError as exc:
            raise ImportError("The 'mlx-lm' package is required for MLXRuntime.") from exc
        except Exception as exc:
            raise RuntimeError(f"MLX inference failed: {exc}") from exc

    def metadata(self) -> dict:
        return {"runtime": "mlx", "model": self._model, "max_tokens": self._max_tokens}

    def shutdown(self) -> None:
        self._model_obj = None
        self._tokenizer = None
        self._loaded = False

    def health_check(self) -> bool:
        try:
            import mlx_lm  # noqa: F401

            return True
        except ImportError:
            return False

    def runtime_metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(
            runtime_name="mlx",
            runtime_version=None,
            backend="mlx",
            execution_mode="local",
            gpu_acceleration=True,
            metadata={
                "model": self._model,
                "max_tokens": self._max_tokens,
            },
        )


def _extract_prompt(task: dict[str, Any]) -> str:
    for key in ("prompt", "question", "problem_statement"):
        val = task.get(key)
        if val and isinstance(val, str):
            return val
    return str(task)
