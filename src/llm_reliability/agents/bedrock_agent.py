from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from llm_reliability.agents.adapters.base_llm_adapter import BaseLLMAdapter
from llm_reliability.agents.adapters.exceptions import (
    AuthenticationError, ProviderError, RateLimitError,
    ResponseValidationError)
from llm_reliability.agents.adapters.provider_registry import ProviderRegistry
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse
from llm_reliability.configs.config import Configuration
from llm_reliability.runtime.provider_base import BaseProvider
from llm_reliability.runtime.registry import RuntimeRegistry

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024


class _BedrockAdapter(BaseLLMAdapter):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._client = None
        self._model = config.metadata.get("model", config.llm) or DEFAULT_MODEL
        self._temperature = float(config.metadata.get("temperature", DEFAULT_TEMPERATURE))
        self._max_tokens = int(config.metadata.get("max_tokens", DEFAULT_MAX_TOKENS))
        self._top_p = float(config.metadata.get("top_p", 1.0))

    def initialize(self) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("The 'boto3' package is required for AWS Bedrock.") from exc
        region = os.environ.get("AWS_REGION", "us-east-1")
        try:
            self._client = boto3.client("bedrock-runtime", region_name=region)
        except Exception as exc:
            raise AuthenticationError(f"AWS Bedrock auth failed: {exc}") from exc
        logger.info("AWS Bedrock client initialised (model=%s, region=%s).", self._model, region)

    def generate(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("_BedrockAdapter.generate() called before initialize().")
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        messages = []
        if request.system_prompt:
            messages.append(
                {
                    "role": "user",
                    "content": f"[System]{request.system_prompt}\n\n{request.prompt}",
                }
            )
        else:
            messages.append({"role": "user", "content": request.prompt})
        body["messages"] = messages

        t0 = time.perf_counter()
        try:
            response = self._client.invoke_model(
                modelId=self._model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "accessdenied" in exc_str or "unauthorized" in exc_str:
                raise AuthenticationError(f"Bedrock access denied: {exc}") from exc
            elif "throttling" in exc_str or "toomanystreams" in exc_str:
                raise RateLimitError(f"Bedrock rate limit: {exc}") from exc
            raise ProviderError(f"Bedrock API error: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = json.loads(response["body"].read())
        content_blocks = result.get("content", [])
        text = " ".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        if not text.strip():
            raise ResponseValidationError("Bedrock returned empty response.")
        usage = result.get("usage", {})
        return LLMResponse(
            text=text,
            finish_reason=result.get("stop_reason", "stop"),
            latency_ms=latency_ms,
            tokens_input=usage.get("input_tokens", 0),
            tokens_output=usage.get("output_tokens", 0),
            model_name=self._model,
            provider="bedrock",
        )

    def shutdown(self) -> None:
        self._client = None

    def provider_metadata(self) -> dict[str, Any]:
        return {"provider": "bedrock", "model": self._model}

    def health_check(self) -> bool:
        return self._client is not None


class BedrockAgent(BaseProvider):
    provider_name: str = "bedrock"
    default_model: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    default_requests_per_second: float = 5.0

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._adapter = _BedrockAdapter(config)

    def initialize(self) -> None:
        self._adapter.initialize()
        self._client = getattr(self._adapter, "_client", None)

    def reset(self) -> None:
        super().reset()
        self._adapter._request_logs.clear()
        self._adapter._response_logs.clear()

    def run(self, task: dict[str, Any]) -> Any:
        prompt = self._extract_prompt(task)
        request = self._build_request(prompt)
        self._rate_limiter.acquire()
        response = self._adapter.retry(
            request, max_attempts=self._max_retries, backoff_seconds=self._retry_backoff
        )
        self._track_cost(response)
        return response.text

    def shutdown(self) -> None:
        self._adapter.shutdown()

    def metadata(self) -> dict[str, Any]:
        base = super().metadata()
        base.update(
            {
                "name": "BedrockAgent",
                "provider": "bedrock",
                "model": self._adapter._model,
            }
        )
        return base

    def _health_check_impl(self) -> bool:
        return self._adapter.health_check()


if not ProviderRegistry.exists("bedrock"):
    ProviderRegistry.register("bedrock", _BedrockAdapter)
if not RuntimeRegistry.exists("bedrock"):
    RuntimeRegistry.register("bedrock", BedrockAgent)
