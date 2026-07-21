"""
Benchmark Contract Tests

Tests whether a Benchmark implementation fulfills its contract.
Includes both positive tests against MockBenchmark and negative tests
against intentionally broken benchmark implementations.
"""

import pytest

from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
from llm_reliability.interfaces.agent import Agent
from llm_reliability.interfaces.benchmark import Benchmark
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord


class DummyAgent(Agent):
    def initialize(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def run(self, task: dict) -> str:
        return "Answer 0"

    def shutdown(self) -> None:
        pass

    def metadata(self) -> dict:
        return {"name": "dummy"}


class BrokenBenchmarkNoInterface(Benchmark):
    """Fails to implement required abstract methods."""
    pass


class BrokenBenchmarkBadOutputs(MockBenchmark):
    """Intentionally breaks output contracts."""
    def get_task(self, task_id: str) -> dict:
        return {}  # Invalid task object

    def run(self, agent, task) -> ExecutionRecord:
        return None  # type: ignore # Violates ExecutionRecord return type


def test_benchmark_implements_interface():
    # Should instantiate correctly
    benchmark = MockBenchmark()
    assert isinstance(benchmark, Benchmark)

    with pytest.raises(TypeError):
        # Should raise TypeError due to missing abstract methods
        BrokenBenchmarkNoInterface()


def test_benchmark_deterministic():
    benchmark1 = MockBenchmark(seed=42)
    benchmark1.load()
    task1 = benchmark1.get_task("mock-task-0")
    record1 = benchmark1.run(DummyAgent(), task1)

    benchmark2 = MockBenchmark(seed=42)
    benchmark2.load()
    task2 = benchmark2.get_task("mock-task-0")
    record2 = benchmark2.run(DummyAgent(), task2)

    assert record1.sha256() == record2.sha256()


def test_benchmark_valid_outputs():
    benchmark = MockBenchmark()
    benchmark.load()
    tasks = benchmark.list_tasks()
    assert isinstance(tasks, list)
    assert len(tasks) > 0

    task = benchmark.get_task(tasks[0])
    assert isinstance(task, dict)
    assert "task_id" in task
    assert task["task_id"] == tasks[0]

    record = benchmark.run(DummyAgent(), task)
    assert isinstance(record, ExecutionRecord)

    eval_record = benchmark.evaluate(record)
    assert isinstance(eval_record, EvaluationRecord)

    meta = benchmark.metadata()
    assert isinstance(meta, dict)

    logs = benchmark.collect_logs()
    assert isinstance(logs, dict)


def test_benchmark_error_handling():
    benchmark = MockBenchmark()
    benchmark.load()
    with pytest.raises(ValueError):
        benchmark.get_task("invalid-task-id")


def test_broken_benchmark_contract():
    benchmark = BrokenBenchmarkBadOutputs()
    benchmark.load()
    task = benchmark.get_task("mock-task-0")
    
    # Contract: get_task should return valid task dict. This broken benchmark returns an empty dict.
    assert "task_id" not in task
    
    record = benchmark.run(DummyAgent(), task)
    # Contract: run should return an ExecutionRecord. This broken benchmark returns None.
    assert not isinstance(record, ExecutionRecord)
