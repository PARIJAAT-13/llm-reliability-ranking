"""Tests for ExperimentOrchestrator."""

import json

import pytest

from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
from llm_reliability.configs.config import Configuration
from llm_reliability.experiments.experiment_models import ExperimentSpec
from llm_reliability.orchestration.experiment_orchestrator import \
    ExperimentOrchestrator
from tests.contracts.test_agent_contract import ValidAgent


@pytest.fixture
def mock_dataset_file(tmp_path):
    path = tmp_path / "mock_dataset.json"
    tasks = [
        {
            "task_id": "mock-0",
            "prompt": "Solve 0",
            "expected_answer": "Answer 0",
            "difficulty": "easy",
            "category": "logic",
        }
    ]
    path.write_text(json.dumps(tasks), encoding="utf-8")
    return path


@pytest.fixture
def sample_yaml_file(tmp_path, mock_dataset_file):
    yaml_content = f"""
name: test_batch
output_dir: {tmp_path.as_posix()}/results
models:
  - MockAgent
  - GPT-4.1
benchmarks:
  - name: MockBenchmark
    dataset_path: {mock_dataset_file.as_posix()}
seeds:
  - 1
  - 2
repetitions: 2
matrix_mode: per_pair
"""
    path = tmp_path / "config.yaml"
    path.write_text(yaml_content, encoding="utf-8")
    return path


@pytest.fixture
def sample_json_file(tmp_path, mock_dataset_file):
    data = {
        "name": "test_json_batch",
        "output_dir": str(tmp_path / "results_json"),
        "models": ["MockAgent"],
        "benchmarks": [{"name": "MockBenchmark", "dataset_path": str(mock_dataset_file)}],
        "seeds": [42],
        "repetitions": 1,
        "matrix_mode": "single",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ── Tests ────────────────────────────────────────────────────────────────────


def test_load_config_file_json_and_yaml(sample_yaml_file, sample_json_file):
    yaml_def = ExperimentOrchestrator.load_config_file(sample_yaml_file)
    assert yaml_def["name"] == "test_batch"
    assert len(yaml_def["models"]) == 2

    json_def = ExperimentOrchestrator.load_config_file(sample_json_file)
    assert json_def["name"] == "test_json_batch"
    assert json_def["matrix_mode"] == "single"


def test_generate_specs_per_pair_matrix():
    definition = {
        "name": "study",
        "models": ["MockAgent", "GPT-4.1"],
        "benchmarks": ["MockBenchmark", "AgentBoard"],
        "seeds": [1, 2],
        "repetitions": 3,
        "matrix_mode": "per_pair",
    }
    specs = ExperimentOrchestrator.generate_specs(definition)
    # 2 benchmarks * 2 models = 4 specs
    assert len(specs) == 4
    for spec in specs:
        assert len(spec.benchmarks) == 1
        assert len(spec.agents) == 1
        assert spec.seeds == [1, 2]
        assert spec.repetitions == 3


def test_generate_specs_per_combination_matrix():
    definition = {
        "name": "study_comb",
        "models": ["m1", "m2"],
        "benchmarks": ["b1"],
        "seeds": [1, 2, 3],
        "repetitions": 1,
        "matrix_mode": "per_combination",
    }
    specs = ExperimentOrchestrator.generate_specs(definition)
    # 1 benchmark * 2 models * 3 seeds = 6 specs
    assert len(specs) == 6


def test_generate_specs_single_matrix():
    definition = {
        "name": "study_single",
        "models": ["m1", "m2"],
        "benchmarks": ["b1", "b2"],
        "seeds": [1, 2],
        "repetitions": 1,
        "matrix_mode": "single",
    }
    specs = ExperimentOrchestrator.generate_specs(definition)
    assert len(specs) == 1
    assert len(specs[0].benchmarks) == 2
    assert len(specs[0].agents) == 2


def test_orchestrator_run_all_success(tmp_path, sample_yaml_file):
    out_dir = tmp_path / "results"
    orchestrator = ExperimentOrchestrator(output_dir=out_dir)

    def benchmark_factory(name: str, config: Configuration):
        return MockBenchmark(seed=config.seed)

    def agent_factory(aspec, config: Configuration):
        return ValidAgent()

    orchestrator._benchmark_factory = benchmark_factory
    orchestrator._agent_factory = agent_factory

    summary = orchestrator.run_from_file(sample_yaml_file)

    assert summary["total_experiments"] == 2
    assert summary["completed_count"] == 2
    assert summary["failed_count"] == 0
    assert (out_dir / "test_batch_master_summary.json").exists()
    assert (out_dir / "test_batch_master_summary.md").exists()


def test_orchestrator_failure_tolerance(tmp_path):
    out_dir = tmp_path / "results_fail"
    orchestrator = ExperimentOrchestrator(output_dir=out_dir)

    # 1 spec that succeeds, 1 spec that fails
    good_spec = ExperimentSpec(
        experiment_name="good_exp",
        benchmarks=[{"name": "MockBenchmark", "dataset_path": "data/mock.json"}],
        agents=[{"name": "MockAgent"}],
        seeds=[1],
        repetitions=1,
    )
    bad_spec = ExperimentSpec(
        experiment_name="bad_exp",
        benchmarks=[{"name": "CrashingBench", "dataset_path": "data/mock.json"}],
        agents=[{"name": "MockAgent"}],
        seeds=[1],
        repetitions=1,
    )

    def benchmark_factory(name: str, config: Configuration):
        if name == "CrashingBench":
            raise RuntimeError("Intentional benchmark failure")
        return MockBenchmark(seed=config.seed)

    orchestrator._benchmark_factory = benchmark_factory

    summary = orchestrator.run_all([good_spec, bad_spec], batch_name="failure_test")

    assert summary["total_experiments"] == 2
    assert summary["completed_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["failed_experiments"][0]["experiment_name"] == "bad_exp"


def test_orchestrator_resume_skips_completed(tmp_path, sample_json_file):
    out_dir = tmp_path / "results_json"
    orchestrator = ExperimentOrchestrator(output_dir=out_dir)

    def benchmark_factory(name: str, config: Configuration):
        return MockBenchmark(seed=config.seed)

    orchestrator._benchmark_factory = benchmark_factory

    # Run once
    summary1 = orchestrator.run_from_file(sample_json_file, resume=True)
    assert summary1["completed_count"] == 1

    # Run again with resume=True — should skip already completed run
    summary2 = orchestrator.run_from_file(sample_json_file, resume=True)
    assert summary2["completed_count"] == 1
    assert summary2["completed_experiments"][0]["status"] == "completed (cached)"


def test_cli_parsing(tmp_path, sample_yaml_file):
    from llm_reliability.orchestration.experiment_orchestrator import main

    out_dir = tmp_path / "cli_results"
    test_args = [
        "experiment_orchestrator.py",
        "--config",
        str(sample_yaml_file),
        "--output-dir",
        str(out_dir),
    ]

    with pytest.MonkeyPatch.context() as m:
        m.setattr("sys.argv", test_args)
        main()

    assert out_dir.exists()
    assert (out_dir / "test_batch_master_summary.json").exists()
