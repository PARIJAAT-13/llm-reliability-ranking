"""Unit tests for multi-runtime adapters (llama.cpp, vLLM, HuggingFace Transformers)."""

import pytest

from llm_reliability.agents.agent_factory import AgentFactory
from llm_reliability.configs.config import Configuration
from llm_reliability.runtime.registry import RuntimeRegistry


@pytest.fixture(autouse=True)
def discover_runtimes():
    """Ensure the RuntimeRegistry is populated before each test."""
    RuntimeRegistry.discover()
    yield


@pytest.fixture
def mock_config():
    return Configuration(
        experiment_name="test_runtime",
        agent="ollama",
        benchmark="GAIA",
        llm="mock",
        prompt_version="v1",
        dataset_version="1.0",
        seed=42,
        repetitions=1,
        metadata={"model": "test-model"},
    )


def test_agent_factory_runtimes(mock_config):
    runtimes = ["ollama", "llamacpp", "vllm", "huggingface", "hf"]
    for rt in runtimes:
        agent = AgentFactory.create(rt, mock_config)
        assert agent is not None
        assert hasattr(agent, "run")
        assert hasattr(agent, "metadata")
