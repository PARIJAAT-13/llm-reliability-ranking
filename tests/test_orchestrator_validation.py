"""
Tests for ExperimentOrchestrator pre-flight matrix validation and dual model configuration syntax parsing.
"""

from unittest.mock import patch
import pytest

from llm_reliability.orchestration.experiment_orchestrator import ExperimentOrchestrator


def test_dual_model_config_parsing_string_and_dict():
    """Verify that generate_specs supports both string and object model configurations."""
    raw_def = {
        "name": "dual_syntax_test",
        "matrix_mode": "per_pair",
        "benchmarks": ["MockBenchmark"],
        "models": [
            "ollama:llama3.1:8b",
            {
                "provider": "ollama",
                "model": "mistral:7b",
                "temperature": 0.2,
                "max_tokens": 512,
            },
            {
                "name": "mock",
            },
        ],
    }

    specs = ExperimentOrchestrator.generate_specs(raw_def)
    assert len(specs) == 3

    # Spec 1: ollama:llama3.1:8b
    assert specs[0].agents[0].name == "ollama"
    assert specs[0].agents[0].agent_metadata["model"] == "llama3.1:8b"

    # Spec 2: dict format with provider and model parameters
    assert specs[1].agents[0].name == "ollama"
    assert specs[1].agents[0].agent_metadata["model"] == "mistral:7b"
    assert specs[1].agents[0].agent_metadata["temperature"] == 0.2
    assert specs[1].agents[0].agent_metadata["max_tokens"] == 512

    # Spec 3: mock format
    assert specs[2].agents[0].name == "mock"


def test_preflight_validation_invalid_benchmark():
    """Verify pre-flight validation raises ValueError for invalid benchmark."""
    raw_def = {
        "name": "invalid_bench_test",
        "benchmarks": ["NonExistentBenchmark"],
        "models": ["mock"],
    }
    specs = ExperimentOrchestrator.generate_specs(raw_def)

    orch = ExperimentOrchestrator()
    with pytest.raises(ValueError, match="is not registered in BenchmarkRegistry"):
        orch.validate_specs(specs, check_ollama_server=False)


def test_preflight_validation_unsupported_provider():
    """Verify pre-flight validation raises ValueError for unsupported agent provider."""
    raw_def = {
        "name": "invalid_provider_test",
        "benchmarks": ["MockBenchmark"],
        "models": ["unsupported_provider_xyz"],
    }
    specs = ExperimentOrchestrator.generate_specs(raw_def)

    orch = ExperimentOrchestrator()
    with pytest.raises(ValueError, match="Unsupported agent provider"):
        orch.validate_specs(specs, check_ollama_server=False)


def test_preflight_validation_ollama_server_check_offline():
    """Verify pre-flight validation fails fast when Ollama server is offline."""
    raw_def = {
        "name": "offline_ollama_test",
        "benchmarks": ["MockBenchmark"],
        "models": ["ollama:llama3.1:8b"],
    }
    specs = ExperimentOrchestrator.generate_specs(raw_def)

    orch = ExperimentOrchestrator()
    with patch("llm_reliability.agents.utils.ollama_utils.check_ollama_server", return_value=(False, "Connection refused")):
        with pytest.raises(ValueError, match="Ollama server reachable check failed"):
            orch.validate_specs(specs, check_ollama_server=True)


def test_preflight_validation_success_mock():
    """Verify pre-flight validation succeeds for valid specs."""
    raw_def = {
        "name": "valid_mock_test",
        "benchmarks": ["MockBenchmark"],
        "models": ["mock"],
    }
    specs = ExperimentOrchestrator.generate_specs(raw_def)
    orch = ExperimentOrchestrator()
    orch.validate_specs(specs, check_ollama_server=False)
