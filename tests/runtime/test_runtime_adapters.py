"""Tests for runtime adapters (Ollama, vLLM, llama.cpp, TGI, LM Studio, MLX, OpenAI-compat)."""

import pytest

from llm_reliability.runtime.adapters import (
    LlamaCppRuntime,
    LMStudioRuntime,
    MLXRuntime,
    OllamaRuntime,
    OpenAICompatRuntime,
    TGIRuntime,
    VLLMRuntime,
)


class TestOllamaRuntime:
    def test_initialization(self):
        runtime = OllamaRuntime(model="test-model", base_url="http://localhost:11434")
        assert runtime._model == "test-model"
        assert runtime._base_url == "http://localhost:11434"

    def test_health_check_not_connected(self):
        runtime = OllamaRuntime()
        assert runtime.health_check() is False

    def test_runtime_metadata(self):
        runtime = OllamaRuntime(model="test")
        meta = runtime.runtime_metadata()
        assert meta.runtime_name == "ollama"
        assert meta.metadata["model"] == "test"


class TestVLLMRuntime:
    def test_initialization(self):
        runtime = VLLMRuntime(model="test-model", base_url="http://localhost:8000/v1")
        assert runtime._model == "test-model"

    def test_health_check_not_connected(self):
        runtime = VLLMRuntime()
        assert runtime.health_check() is False

    def test_runtime_metadata(self):
        runtime = VLLMRuntime(model="test")
        meta = runtime.runtime_metadata()
        assert meta.runtime_name == "vllm"
        assert meta.backend == "vLLM"


class TestLlamaCppRuntime:
    def test_initialization(self):
        runtime = LlamaCppRuntime(model="test", base_url="http://localhost:8080/completion")
        assert runtime._model == "test"

    def test_health_check_not_connected(self):
        runtime = LlamaCppRuntime()
        assert runtime.health_check() is False

    def test_runtime_metadata(self):
        runtime = LlamaCppRuntime(model="test")
        meta = runtime.runtime_metadata()
        assert meta.runtime_name == "llama.cpp"


class TestTGIRuntime:
    def test_initialization(self):
        runtime = TGIRuntime(model="test", base_url="http://localhost:8080/v1")
        assert runtime._model == "test"

    def test_health_check_not_connected(self):
        runtime = TGIRuntime()
        assert runtime.health_check() is False

    def test_runtime_metadata(self):
        runtime = TGIRuntime(model="test")
        meta = runtime.runtime_metadata()
        assert meta.runtime_name == "tgi"
        assert meta.backend == "huggingface-tgi"


class TestLMStudioRuntime:
    def test_initialization(self):
        runtime = LMStudioRuntime(model="test", base_url="http://localhost:1234/v1")
        assert runtime._model == "test"

    def test_health_check_not_connected(self):
        runtime = LMStudioRuntime()
        assert runtime.health_check() is False

    def test_runtime_metadata(self):
        runtime = LMStudioRuntime(model="test")
        meta = runtime.runtime_metadata()
        assert meta.runtime_name == "lm-studio"


class TestMLXRuntime:
    def test_initialization(self):
        runtime = MLXRuntime(model="test-model")
        assert runtime._model == "test-model"

    def test_health_check_no_mlx(self):
        runtime = MLXRuntime()
        # mlx-lm not installed in test env, so health_check should return False
        assert runtime.health_check() is False

    def test_runtime_metadata(self):
        runtime = MLXRuntime(model="test")
        meta = runtime.runtime_metadata()
        assert meta.runtime_name == "mlx"
        assert meta.backend == "mlx"


class TestOpenAICompatRuntime:
    def test_initialization(self):
        runtime = OpenAICompatRuntime(model="gpt-4o", base_url="https://api.openai.com/v1")
        assert runtime._model == "gpt-4o"

    def test_health_check_not_connected(self):
        runtime = OpenAICompatRuntime()
        assert runtime.health_check() is False

    def test_runtime_metadata(self):
        runtime = OpenAICompatRuntime(model="gpt-4o")
        meta = runtime.runtime_metadata()
        assert meta.runtime_name == "openai-compat"
        assert meta.execution_mode == "api"


class TestRuntimeAdapterRegistration:
    def test_all_adapters_accessible(self):
        # Verify all classes exist and are Runtime subclasses
        from llm_reliability.runtime import Runtime
        from llm_reliability.runtime.adapters import (
            LlamaCppRuntime,
            LMStudioRuntime,
            MLXRuntime,
            OllamaRuntime,
            OpenAICompatRuntime,
            TGIRuntime,
            VLLMRuntime,
        )

        adapters = [
            OllamaRuntime,
            VLLMRuntime,
            LlamaCppRuntime,
            TGIRuntime,
            LMStudioRuntime,
            MLXRuntime,
            OpenAICompatRuntime,
        ]
        for adapter in adapters:
            assert issubclass(adapter, Runtime), f"{adapter.__name__} is not a Runtime subclass"
