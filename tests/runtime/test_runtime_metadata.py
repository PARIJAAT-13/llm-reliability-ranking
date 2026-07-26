"""Tests for runtime metadata models and enhanced Runtime interface."""

import pytest

from llm_reliability.runtime import (Runtime, RuntimeCapabilities,
                                     RuntimeMetadata, RuntimeRegistry)


class TestRuntimeMetadata:
    def test_minimal_creation(self) -> None:
        meta = RuntimeMetadata(runtime_name="test")
        assert meta.runtime_name == "test"
        assert meta.runtime_version is None
        assert meta.gpu_acceleration is False
        assert meta.inference_parameters == {}

    def test_full_creation(self) -> None:
        meta = RuntimeMetadata(
            runtime_name="ollama",
            runtime_version="0.1.30",
            backend="ollama",
            api_version="v1",
            execution_mode="local",
            quantization="q4_k_m",
            context_length=8192,
            gpu_acceleration=True,
            gpu_devices=["NVIDIA RTX 4090"],
            thread_count=8,
            batch_size=1,
            inference_parameters={"temperature": 0.0, "max_tokens": 1024},
        )
        assert meta.runtime_name == "ollama"
        assert meta.quantization == "q4_k_m"
        assert meta.context_length == 8192

    def test_capabilities_defaults(self) -> None:
        caps = RuntimeCapabilities()
        assert caps.model_loading is False
        assert caps.health_check is False
        assert caps.token_counting is False

    def test_capabilities_custom(self) -> None:
        caps = RuntimeCapabilities(model_loading=True, health_check=True)
        assert caps.model_loading is True
        assert caps.health_check is True
        assert caps.token_counting is False

    def test_serialization_round_trip(self) -> None:
        meta = RuntimeMetadata(
            runtime_name="test",
            runtime_version="1.0",
            gpu_acceleration=True,
        )
        json_str = meta.canonical_json()
        restored = RuntimeMetadata.from_canonical_json(json_str)
        assert meta == restored


class TestEnhancedRuntimeInterface:
    def test_default_implementations(self) -> None:
        runtime = _create_minimal_runtime()
        assert runtime.health_check() is True
        assert runtime.count_tokens("hello") == 0
        assert runtime.measure_memory() == {}
        assert runtime.load_model() is None
        assert runtime.unload_model() is None

    def test_execute_delegates_to_run(self) -> None:
        runtime = _create_minimal_runtime()
        result = runtime.execute({"prompt": "hello"})
        assert result == "run:hello"

    def test_measure_latency(self) -> None:
        runtime = _create_minimal_runtime()
        output, latency_ms = runtime.measure_latency({"prompt": "hello"})
        assert output == "run:hello"
        assert latency_ms >= 0.0

    def test_runtime_metadata_default(self) -> None:
        runtime = _create_minimal_runtime()
        meta = runtime.runtime_metadata()
        assert meta.runtime_name == "_MinimalTestRuntime"
        assert meta.runtime_version is None

    def test_detect_capabilities(self) -> None:
        runtime = _create_minimal_runtime()
        caps = runtime._detect_capabilities()
        assert caps.model_loading is False
        assert caps.health_check is False

    def test_capability_override_detection(self) -> None:
        runtime = _create_capable_runtime()
        caps = runtime._detect_capabilities()
        assert caps.health_check is True
        assert caps.model_loading is True


class TestRuntimeRegistry:
    def test_register_and_list(self) -> None:
        RuntimeRegistry.register("test-rr", _MinimalTestRuntime)
        names = RuntimeRegistry.list()
        assert "test-rr" in names
        RuntimeRegistry.unregister("test-rr")


class _MinimalTestRuntime(Runtime):
    def initialize(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def run(self, task: dict) -> str:
        return f"run:{task.get('prompt', '')}"

    def shutdown(self) -> None:
        pass

    def metadata(self) -> dict:
        return {"name": "MinimalTestRuntime"}


class _CapableTestRuntime(Runtime):
    def initialize(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def run(self, task: dict) -> str:
        return f"run:{task.get('prompt', '')}"

    def shutdown(self) -> None:
        pass

    def metadata(self) -> dict:
        return {"name": "CapableTestRuntime"}

    def health_check(self) -> bool:
        return True

    def load_model(self) -> None:
        pass


def _create_minimal_runtime() -> Runtime:
    return _MinimalTestRuntime()


def _create_capable_runtime() -> Runtime:
    return _CapableTestRuntime()
