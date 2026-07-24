"""
Tests for OllamaAgent and Ollama local model integration.
"""

from unittest.mock import MagicMock, patch

from llm_reliability.agents.agent_factory import AgentFactory
from llm_reliability.agents.ollama_agent import OllamaAgent, _OllamaAdapter
from llm_reliability.configs.config import Configuration
from llm_reliability.experiments.experiment_models import (
    AgentSpec,
    BenchmarkSpec,
    ExperimentSpec,
)
from llm_reliability.experiments.experiment_runner import ExperimentRunner


def make_config(**kwargs) -> Configuration:
    defaults = {
        "experiment_name": "test",
        "benchmark": "mock",
        "agent": "ollama",
        "llm": "mock",
        "prompt_version": "1",
        "dataset_version": "1",
        "seed": 42,
        "repetitions": 1,
    }
    defaults.update(kwargs)
    return Configuration(**defaults)


def test_ollama_agent_factory_resolution():
    """Verify that AgentFactory resolves 'ollama' to OllamaAgent."""
    cfg = make_config(agent="ollama", metadata={"model": "mistral:7b"})
    agent = AgentFactory.create("ollama", cfg)
    assert isinstance(agent, OllamaAgent)
    assert agent._adapter._model == "mistral:7b"


def test_ollama_adapter_default_model():
    """Verify fallback default model when model metadata is missing."""
    cfg = make_config(agent="ollama")
    adapter = _OllamaAdapter(cfg)
    assert adapter._model == "llama3.1:8b"


def test_ollama_adapter_initialization(caplog):
    """Verify OpenAI client initialization with Ollama base URL and log output."""
    cfg = make_config(agent="ollama", metadata={"model": "qwen2.5:7b"})
    adapter = _OllamaAdapter(cfg)

    with patch("openai.OpenAI") as mock_openai_cls, caplog.at_level("INFO"):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        adapter.initialize()

        mock_openai_cls.assert_called_once_with(
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama",
        )
        assert "Initializing OllamaAgent" in caplog.text
        assert "Model: qwen2.5:7b" in caplog.text


def test_experiment_spec_multiple_ollama_models():
    """Verify ExperimentSpec allows multiple ollama models without raising duplicate agent error."""
    agents = [
        AgentSpec(name="ollama", metadata={"model": "llama3.1:8b"}),
        AgentSpec(name="ollama", metadata={"model": "mistral:7b"}),
        AgentSpec(name="ollama", metadata={"model": "qwen2.5:7b"}),
        AgentSpec(name="ollama", metadata={"model": "gemma2:9b"}),
    ]
    spec = ExperimentSpec(
        experiment_name="ollama_multi_test",
        benchmarks=[BenchmarkSpec(name="mock", dataset_path="data/mock.json")],
        agents=agents,
        seeds=[42],
    )
    assert len(spec.agents) == 4


def test_experiment_runner_build_config_propagates_model():
    """Verify ExperimentRunner._build_config passes model metadata to Configuration."""
    agents = [
        AgentSpec(name="ollama", metadata={"model": "gemma2:9b"}),
    ]
    spec = ExperimentSpec(
        experiment_name="config_prop_test",
        benchmarks=[BenchmarkSpec(name="mock", dataset_path="data/mock.json")],
        agents=agents,
        seeds=[42],
    )
    runner = ExperimentRunner(spec=spec, agent_factory=lambda spec, cfg: MagicMock())
    queue = runner._scheduler.build_run_queue()
    assert len(queue) == 1
    assert queue[0].agent_name == "ollama:gemma2:9b"

    cfg = runner._build_config(queue[0])
    assert cfg.metadata.get("model") == "gemma2:9b"
    assert cfg.metadata.get("dataset_path") == "data/mock.json"


def test_ollama_non_retryable_exceptions():
    """Verify that deterministic Ollama errors have is_transient=False and fail immediately without retry."""
    from llm_reliability.agents.adapters.exceptions import (
        AuthenticationError,
        ConnectionError,
        OllamaMemoryError,
        OllamaModelNotFoundError,
        OllamaServerNotFoundError,
        RateLimitError,
        RequestValidationError,
        ResponseValidationError,
    )

    assert OllamaModelNotFoundError.is_transient is False
    assert OllamaMemoryError.is_transient is False
    assert OllamaServerNotFoundError.is_transient is False
    assert AuthenticationError.is_transient is False
    assert RequestValidationError.is_transient is False
    assert ResponseValidationError.is_transient is False

    assert RateLimitError.is_transient is True
    assert ConnectionError.is_transient is True


def test_ollama_shutdown_unloads_model():
    """Verify that shutting down OllamaAgent triggers keep_alive=0 model unloading."""
    cfg = make_config(agent="ollama", metadata={"model": "phi3:mini"})
    agent = AgentFactory.create("ollama", cfg)

    with patch("llm_reliability.agents.ollama_agent.unload_ollama_model") as mock_unload:
        agent.shutdown()
        mock_unload.assert_called_once_with("phi3:mini", "http://127.0.0.1:11434")


def test_ollama_utils_diagnostics_formatting():
    """Verify error message formatting functions produce actionable messages."""
    from llm_reliability.agents.utils.ollama_utils import (
        format_memory_error,
        format_model_not_found_error,
        model_matches,
    )

    err_msg = format_model_not_found_error(["llama3.1:8b"], ["phi3:mini", "mistral:7b"])
    assert "llama3.1:8b" in err_msg
    assert "ollama pull" in err_msg

    mem_msg = format_memory_error("llama3.1:70b", 40.0, 16.0)
    assert "40.0 GB" in mem_msg
    assert "16.0 GB" in mem_msg

    assert model_matches("llama3.1:8b", ["llama3.1:8b:latest"])
    assert model_matches("mistral", ["mistral:latest"])
    assert not model_matches("nonexistent", ["phi3:mini"])


def test_ollama_memory_error_skips_remaining_tasks(caplog):
    """Verify that when Ollama raises a memory allocation error, the pipeline logs the model skip message, skips remaining tasks, and produces valid evaluation/metric artifacts."""
    from llm_reliability.agents.adapters.exceptions import OllamaMemoryError
    from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
    from llm_reliability.pipeline.experiment_pipeline import ExperimentPipeline

    cfg = make_config(benchmark="MockBenchmark", agent="ollama", metadata={"model": "llama3.1:8b"})
    bench = MockBenchmark(config=cfg)
    agent = MagicMock()
    agent.run.side_effect = OllamaMemoryError(
        "Ollama memory error: model requires 26 GiB of system memory"
    )

    pipeline = ExperimentPipeline(config=cfg, benchmark=bench, agent=agent)

    with caplog.at_level("INFO"):
        result = pipeline.run()

    # Log verification
    assert "Model llama3.1:8b skipped." in caplog.text
    assert "Reason: insufficient system memory." in caplog.text
    assert "Continuing with next scheduled model." in caplog.text

    # Artifact verification
    assert len(result.execution_records) == 10
    assert all(rec.status == "error" for rec in result.execution_records)
    assert result.execution_records[0].environment_metadata.get("failure_reason") == "memory"
    assert len(result.evaluation_records) == 10
    assert len(result.metric_records) > 0
    assert len(result.ranking_records) > 0
