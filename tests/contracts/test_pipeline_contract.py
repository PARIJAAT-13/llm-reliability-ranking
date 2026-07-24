"""
Pipeline Contract Tests

Tests whether the ExperimentPipeline orchestrates execution correctly
and fulfills its contract regardless of the underlying benchmark or agent.
"""

import pytest

from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
from llm_reliability.configs.config import Configuration
from llm_reliability.pipeline.experiment_pipeline import (
    ExperimentPipeline,
    ExperimentResult,
)
from llm_reliability.records.execution import ExecutionRecord
from tests.contracts.test_agent_contract import ValidAgent
from tests.contracts.test_benchmark_contract import DummyAgent


@pytest.fixture
def config():
    return Configuration(
        experiment_name="test_pipeline_contract",
        benchmark="mock",
        agent="valid",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
    )


def test_pipeline_orchestration(config):
    benchmark = MockBenchmark(seed=42)
    agent = ValidAgent()
    pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=agent)

    result = pipeline.run()

    assert isinstance(result, ExperimentResult)
    # Pipeline Contract:
    # ✓ Execute successfully
    # ✓ Generate ExecutionRecords
    # ✓ Generate EvaluationRecords
    # ✓ Generate MetricRecords
    # ✓ Generate RankingRecords
    assert len(result.execution_records) > 0
    assert len(result.evaluation_records) > 0
    assert len(result.metric_records) > 0
    assert len(result.ranking_records) > 0


class CrashingBenchmark(MockBenchmark):
    def run(self, agent: ValidAgent, task: dict) -> "ExecutionRecord":
        raise RuntimeError("Benchmark crash")


def test_pipeline_failure_recovery(config):
    benchmark = CrashingBenchmark(seed=42)
    agent = ValidAgent()
    pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=agent)

    # Contract: Handle failures gracefully (won't crash the pipeline)
    result = pipeline.run()

    assert len(result.metadata["errors"]) > 0
    assert "Benchmark crash" in str(result.metadata["errors"])


def test_pipeline_repeatability(config):
    benchmark1 = MockBenchmark(seed=42)
    pipeline1 = ExperimentPipeline(config=config, benchmark=benchmark1, agent=DummyAgent())
    res1 = pipeline1.run()

    benchmark2 = MockBenchmark(seed=42)
    pipeline2 = ExperimentPipeline(config=config, benchmark=benchmark2, agent=DummyAgent())
    res2 = pipeline2.run()

    # Check that executions are entirely deterministic
    for e1, e2 in zip(res1.execution_records, res2.execution_records):
        assert e1.sha256() == e2.sha256()


def test_pipeline_accepts_any_valid_components(config):
    # Pipeline contract dictates it must accept any valid Benchmark and Agent
    benchmark = MockBenchmark()
    agent = DummyAgent()

    pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=agent)
    assert pipeline.benchmark is benchmark
    assert pipeline.agent is agent
