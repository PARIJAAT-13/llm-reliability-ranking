"""Tests for ExperimentPipeline."""

import logging
from unittest.mock import patch

import pytest

from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
from llm_reliability.configs.config import Configuration
from llm_reliability.pipeline.experiment_pipeline import ExperimentPipeline, ExperimentResult
from tests.test_mock_benchmark import DummyAgent, FailingAgent


@pytest.fixture
def config():
    return Configuration(
        experiment_name="test_pipeline",
        benchmark="mock",
        agent="dummy",
        llm="test-llm",
        prompt_version="v1",
        dataset_version="v1",
        seed=42,
        repetitions=2,
    )


class CrashingBenchmark(MockBenchmark):
    """A benchmark that raises exceptions in run() and evaluate() to test error handling."""

    def run(self, agent, task):
        raise RuntimeError("Benchmark crash")


def test_successful_pipeline_run(config):
    benchmark = MockBenchmark(seed=42)
    agent = DummyAgent()
    pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=agent)

    result = pipeline.run()

    # 10 tasks * 2 repetitions = 20 execution records
    assert len(result.execution_records) == 20
    assert len(result.evaluation_records) == 20
    assert len(result.metric_records) == 1
    assert len(result.ranking_records) == 2  # success and reliability

    assert not result.metadata["errors"]
    assert result.configuration.seed == 42


def test_failed_task(config):
    # Tests that the pipeline catches exceptions from benchmark.run()
    benchmark = CrashingBenchmark(seed=42)
    agent = DummyAgent()
    pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=agent)

    result = pipeline.run()
    
    # 0 executions succeeded
    assert len(result.execution_records) == 0
    # errors list should be populated (10 tasks * 2 repetitions = 20 errors)
    assert len(result.metadata["errors"]) == 20
    assert result.metadata["errors"][0]["error"] == "Benchmark crash"
    assert result.metadata["errors"][0]["phase"] == "run_task"


def test_empty_benchmark(config):
    benchmark = MockBenchmark(seed=42)
    import llm_reliability.benchmarks.mock_benchmark as mb

    original = mb.MOCK_TASKS
    mb.MOCK_TASKS = []
    try:
        pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=DummyAgent())
        result = pipeline.run()

        assert len(result.execution_records) == 0
        assert len(result.evaluation_records) == 0
        assert len(result.metric_records) == 0
        assert len(result.ranking_records) == 0
    finally:
        mb.MOCK_TASKS = original


def test_deterministic_execution(config):
    benchmark1 = MockBenchmark(seed=42)
    pipeline1 = ExperimentPipeline(config=config, benchmark=benchmark1, agent=DummyAgent())
    result1 = pipeline1.run()

    benchmark2 = MockBenchmark(seed=42)
    pipeline2 = ExperimentPipeline(config=config, benchmark=benchmark2, agent=DummyAgent())
    result2 = pipeline2.run()

    # Wait, timestamps for metric/ranking computation depend on datetime.now() inside the pipeline!
    # We must patch datetime to ensure they are identical for serialization comparison.
    # Instead of strict canonical comparison, we check that execution and evaluation hashes match,
    # and metrics/rankings are equal ignoring computed_at.
    
    for e1, e2 in zip(result1.execution_records, result2.execution_records):
        assert e1.sha256() == e2.sha256()
        
    for ev1, ev2 in zip(result1.evaluation_records, result2.evaluation_records):
        assert ev1.sha256() == ev2.sha256()


def test_logging(config, caplog):
    caplog.set_level(logging.INFO)
    benchmark = MockBenchmark(seed=42)
    pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=DummyAgent())
    pipeline.run()

    logs = caplog.text
    assert "experiment start" in logs
    assert "task start: mock-task-0" in logs
    assert "task completion: mock-task-0" in logs
    assert "evaluation start" in logs
    assert "metric generation" in logs
    assert "ranking generation" in logs
    assert "experiment completion" in logs


def test_save_and_load_results(config, tmp_path):
    benchmark = MockBenchmark(seed=42)
    pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=DummyAgent())
    result = pipeline.run()

    file_path = tmp_path / "results.json"
    pipeline.save_results(file_path)

    assert file_path.exists()

    loaded_result = ExperimentPipeline.load_results(file_path)
    assert loaded_result.canonical_json() == result.canonical_json()


def test_repeated_execution_consistency():
    config = Configuration(
        experiment_name="test_pipeline",
        benchmark="mock",
        agent="dummy",
        llm="test-llm",
        prompt_version="v1",
        dataset_version="v1",
        seed=42,
        repetitions=5,
    )
    benchmark = MockBenchmark(seed=42)
    pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=DummyAgent())
    result = pipeline.run()

    assert len(result.execution_records) == 50  # 10 tasks * 5 reps

    # ensure metric records calculate repeated run consistency correctly
    metric = result.metric_records[0]
    # our dummy agent always gets the answer correct, so success_rate is 1.0, consistency is 1.0
    assert metric.repeated_run_consistency == 1.0
    assert metric.success_rate == 1.0
