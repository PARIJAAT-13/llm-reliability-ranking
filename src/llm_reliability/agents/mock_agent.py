"""Mock agent implementation for testing and development."""

from __future__ import annotations

from typing import Any

from llm_reliability.agents.adapters.base_llm_adapter import BaseLLMAdapter
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse
from llm_reliability.configs.config import Configuration
from llm_reliability.runtime.provider_base import BaseProvider
from llm_reliability.runtime.registry import RuntimeRegistry


class _MockAdapter(BaseLLMAdapter):
    _client: Any = None
    _model: str = "mock-model"

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._client = None

    def initialize(self) -> None:
        self._client = "mock_client"

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=request.prompt,
            finish_reason="stop",
            latency_ms=0.0,
            tokens_input=len(request.prompt.split()),
            tokens_output=len(request.prompt.split()),
            model_name="mock-model",
            provider="mock",
        )

    def shutdown(self) -> None:
        self._client = None

    def provider_metadata(self) -> dict[str, Any]:
        return {"provider": "mock", "model": "mock-model"}

    def health_check(self) -> bool:
        return True


class MockAgent(BaseProvider):
    provider_name: str = "mock"
    default_model: str = "mock-model"
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    default_requests_per_second: float = 100.0

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._adapter = _MockAdapter(config)

    def initialize(self) -> None:
        self._adapter.initialize()
        self._client = getattr(self._adapter, "_client", None)

    def reset(self) -> None:
        super().reset()
        self._adapter._request_logs.clear()
        self._adapter._response_logs.clear()

    def run(self, task: dict[str, Any]) -> Any:
        prompt = task.get("prompt") or task.get("expected_answer") or str(task)
        request = self._build_request(prompt)
        response = self._adapter.generate(request)
        self._track_cost(response)
        return task.get("expected_answer", response.text)

    def shutdown(self) -> None:
        self._adapter.shutdown()

    def metadata(self) -> dict[str, Any]:
        base = super().metadata()
        base.update({"name": "MockAgent", "provider": "mock", "model": self._adapter._model})
        return base

    def _health_check_impl(self) -> bool:
        return self._adapter.health_check()


if not RuntimeRegistry.exists("mock"):
    RuntimeRegistry.register("mock", MockAgent)
