"""Tests for RepeatedRunner."""

import logging

import pytest

from llm_reliability.agents.mock_agent import MockAgent
from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
from llm_reliability.configs.config import Configuration
from llm_reliability.metrics.consistency import compute_consistency
from llm_reliability.records.metric import MetricRecord
from llm_reliability.reliability.repeated_runner import (
    RepeatedRunner,
    RepeatedRunResult,
)
from tests.contracts.test_agent_contract import ValidAgent


@pytest.fixture
def config():
    return Configuration(
        experiment_name="test_repeated_runner",
        benchmark="MockBenchmark",
        agent="MockAgent",
        llm="mock",
        prompt_version="v1",
        dataset_version="v1",
        seed=42,
        repetitions=5,
    )


@pytest.fixture
def benchmark(config):
    b = MockBenchmark(config=config)
    b.load()
    return b


@pytest.fixture
def agent(config):
    return MockAgent(config=config)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_repeated_runner_instantiation(config, benchmark, agent):
    runner = RepeatedRunner(config=config, benchmark=benchmark, agent=agent)
    assert runner.config == config
    assert runner.benchmark == benchmark
    assert runner.agent == agent


def test_repeated_run_success_path(config, benchmark, agent):
    runner = RepeatedRunner(config=config, benchmark=benchmark, agent=agent)
    task = benchmark.get_task("mock-task-0")

    result = runner.run_repeated_task(task=task, repetitions=5)

    assert isinstance(result, RepeatedRunResult)
    assert result.task_id == "mock-task-0"
    assert result.repetitions == 5
    assert len(result.execution_records) == 5
    assert len(result.evaluation_records) == 5
    assert len(result.errors) == 0

    # Ensure run_index values are 0, 1, 2, 3, 4
    run_indices = [e.run_index for e in result.execution_records]
    assert run_indices == [0, 1, 2, 3, 4]

    eval_run_indices = [ev.run_index for ev in result.evaluation_records]
    assert eval_run_indices == [0, 1, 2, 3, 4]

    # Ensure execution hashes are unique per repetition
    hashes = [e.sha256() for e in result.execution_records]
    assert len(set(hashes)) == 5


def test_repeated_run_preserves_telemetry_and_outputs(config, benchmark, agent):
    runner = RepeatedRunner(config=config, benchmark=benchmark, agent=agent)
    task = benchmark.get_task("mock-task-1")

    result = runner.run_repeated_task(task=task, repetitions=3)

    for ex in result.execution_records:
        assert ex.runtime_seconds > 0.0
        assert ex.timestamp is not None
        assert ex.status == "success"

    for ev in result.evaluation_records:
        assert ev.score == 1.0
        assert ev.success is True


class ResetCounterAgent(ValidAgent):
    """Agent that counts how many times reset() was called."""

    def __init__(self):
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def test_repeated_run_resets_agent_between_runs(config, benchmark):
    counting_agent = ResetCounterAgent()
    runner = RepeatedRunner(config=config, benchmark=benchmark, agent=counting_agent)
    task = benchmark.get_task("mock-task-0")

    runner.run_repeated_task(task=task, repetitions=4)

    assert counting_agent.reset_count == 4


class IntermittentFailingAgent(ValidAgent):
    """Agent that fails on specific run attempts."""

    def __init__(self):
        self.call_count = 0

    def run(self, task):
        self.call_count += 1
        if self.call_count == 2:
            raise RuntimeError("Intermittent agent failure")
        return "Answer 0"


def test_repeated_run_error_handling_continues_execution(config, benchmark):
    failing_agent = IntermittentFailingAgent()
    runner = RepeatedRunner(config=config, benchmark=benchmark, agent=failing_agent)
    task = benchmark.get_task("mock-task-0")

    result = runner.run_repeated_task(task=task, repetitions=4)

    # Should attempt all 4 repetitions
    assert len(result.execution_records) == 4
    assert len(result.evaluation_records) == 4
    assert len(result.errors) == 1
    assert result.errors[0]["run_index"] == 1
    assert "Intermittent agent failure" in result.errors[0]["error"]

    # Repetition 1 failed, repetitions 0, 2, 3 succeeded
    assert result.execution_records[1].status == "error"
    assert result.execution_records[0].status == "success"
    assert result.execution_records[2].status == "success"


def test_repeated_run_logging(config, benchmark, agent, caplog):
    caplog.set_level(logging.INFO)
    runner = RepeatedRunner(config=config, benchmark=benchmark, agent=agent)
    task = benchmark.get_task("mock-task-0")

    runner.run_repeated_task(task=task, repetitions=3)

    log_text = caplog.text
    assert "Run 1/3" in log_text
    assert "Run 2/3" in log_text
    assert "Run 3/3" in log_text
    assert "completed in" in log_text


def test_repeated_run_compatibility_with_metrics(config, benchmark, agent):
    runner = RepeatedRunner(config=config, benchmark=benchmark, agent=agent)
    task = benchmark.get_task("mock-task-0")

    result = runner.run_repeated_task(task=task, repetitions=5)

    # Compute repeated-run consistency directly via compute_consistency()
    consistency = compute_consistency(result.evaluation_records)
    assert consistency == 1.0

    # Compute MetricRecord from evaluation records
    metric = MetricRecord.from_evaluations(
        result.evaluation_records, computed_at="2026-01-01T00:00:00+00:00"
    )
    assert metric.repeated_run_consistency == 1.0
    assert metric.success_rate == 1.0
