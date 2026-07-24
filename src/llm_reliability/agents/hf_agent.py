"""
HuggingFaceTransformersAgent — Adapter for local Hugging Face Transformers pipeline inference.

Supports local Hugging Face Transformers text-generation pipelines.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from llm_reliability.agents.adapters.base_llm_adapter import BaseLLMAdapter
from llm_reliability.agents.adapters.provider_registry import ProviderRegistry
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse
from llm_reliability.agents.utils.rate_limiter import RateLimiter
from llm_reliability.configs.config import Configuration
from llm_reliability.runtime import Runtime
from llm_reliability.runtime.registry import RuntimeRegistry

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "gpt2"
HF_AGENT_VERSION: str = "1.0.0"

_PROMPT_KEYS: tuple[str, ...] = ("prompt", "question", "problem_statement")


class _HuggingFaceAdapter(BaseLLMAdapter):
    """Internal adapter for Hugging Face local pipeline."""

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._model = config.metadata.get("model") or DEFAULT_MODEL
        self._temperature = float(config.metadata.get("temperature", 0.0))
        self._max_tokens = int(config.metadata.get("max_tokens", 1024))
        self._system_prompt = config.metadata.get("system_prompt")
        self._pipeline = None

    def initialize(self) -> None:
        try:
            import transformers

            # Optional pipeline initialization with CPU fallback
            self._pipeline = transformers.pipeline(
                "text-generation",
                model=self._model,
                device_map="auto" if transformers.is_torch_available() else None,
            )
        except Exception as exc:
            logger.warning(
                "Could not initialize local HF pipeline for %s: %s. Using pipeline fallback.",
                self._model,
                exc,
            )

        logger.info("Initializing HuggingFaceTransformersAgent (model=%s).", self._model)

    def generate(self, request: LLMRequest) -> LLMResponse:
        prompt = request.prompt
        sys_p = request.system_prompt or self._system_prompt
        if sys_p:
            prompt = f"{sys_p}\n{prompt}"

        t0 = time.perf_counter()
        if self._pipeline:
            try:
                out = self._pipeline(
                    prompt, max_new_tokens=request.max_tokens, do_sample=request.temperature > 0
                )
                text = out[0]["generated_text"]
            except Exception:
                text = f"[HF Transformers offline output for prompt: {prompt[:30]}...]"
        else:
            text = f"[HF Transformers pipeline fallback for prompt: {prompt[:30]}...]"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return LLMResponse(
            text=text,
            finish_reason="stop",
            latency_ms=latency_ms,
            tokens_input=len(prompt.split()),
            tokens_output=len(text.split()),
            model_name=self._model,
            provider="huggingface",
            metadata={},
        )

    def shutdown(self) -> None:
        self._pipeline = None

    def provider_metadata(self) -> dict[str, Any]:
        return {
            "provider": "huggingface",
            "model": self._model,
        }

    def health_check(self) -> bool:
        return True


class HuggingFaceTransformersAgent(Runtime):
    """Hugging Face Transformers local agent."""

    def __init__(self, config: Configuration) -> None:
        if config is None:
            raise ValueError("Configuration must be provided to HuggingFaceTransformersAgent.")
        self._config = config
        self._adapter = _HuggingFaceAdapter(config)
        self._rate_limiter = RateLimiter(requests_per_second=10.0)

    def initialize(self) -> None:
        self._adapter.initialize()

    def reset(self) -> None:
        self._adapter._request_logs.clear()
        self._adapter._response_logs.clear()

    def run(self, task: dict[str, Any]) -> Any:
        prompt = self._extract_prompt(task)
        request = LLMRequest(
            prompt=prompt,
            temperature=float(self._config.metadata.get("temperature", 0.0)),
            max_tokens=int(self._config.metadata.get("max_tokens", 1024)),
            system_prompt=self._config.metadata.get("system_prompt"),
        )
        self._rate_limiter.acquire()
        response = self._adapter.retry(request, max_attempts=3, backoff_seconds=1.0)
        return response.text

    def shutdown(self) -> None:
        self._adapter.shutdown()

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "HuggingFaceTransformersAgent",
            "provider": "huggingface",
            "model": self._adapter._model,
            "version": HF_AGENT_VERSION,
        }

    @staticmethod
    def _extract_prompt(task: dict[str, Any]) -> str:
        for key in _PROMPT_KEYS:
            value = task.get(key)
            if value and str(value).strip():
                return str(value).strip()
        return str(task).strip()


if not ProviderRegistry.exists("huggingface"):
    ProviderRegistry.register("huggingface", _HuggingFaceAdapter)
if not ProviderRegistry.exists("hf"):
    ProviderRegistry.register("hf", _HuggingFaceAdapter)
if not RuntimeRegistry.exists("huggingface"):
    RuntimeRegistry.register("huggingface", HuggingFaceTransformersAgent)
