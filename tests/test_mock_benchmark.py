"""Tests for MockBenchmark."""

import pytest

from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
from llm_reliability.interfaces.agent import Agent


class DummyAgent(Agent):
    """Dummy agent for testing."""

    def initialize(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def run(self, task: dict) -> str:
        return task["expected_answer"]

    def shutdown(self) -> None:
        pass

    def metadata(self) -> dict:
        return {"name": "dummy"}


class FailingAgent(Agent):
    """Failing agent for testing."""

    def initialize(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def run(self, task: dict) -> str:
        raise ValueError("Agent failed")

    def shutdown(self) -> None:
        pass

    def metadata(self) -> dict:
        return {"name": "failing"}


def test_load_and_list_tasks():
    benchmark = MockBenchmark()
    benchmark.load()
    tasks = benchmark.list_tasks()
    assert len(tasks) == 10
    assert tasks[0] == "mock-task-0"


def test_list_tasks_before_load():
    benchmark = MockBenchmark()
    with pytest.raises(RuntimeError, match="Benchmark not loaded."):
        benchmark.list_tasks()


def test_get_task():
    benchmark = MockBenchmark()
    benchmark.load()
    task = benchmark.get_task("mock-task-0")
    assert task["task_id"] == "mock-task-0"
    assert task["expected_answer"] == "Answer 0"


def test_get_task_invalid_id():
    benchmark = MockBenchmark()
    benchmark.load()
    with pytest.raises(ValueError, match="Unknown task_id: invalid"):
        benchmark.get_task("invalid")


def test_run_success():
    benchmark = MockBenchmark(seed=42)
    benchmark.load()
    task = benchmark.get_task("mock-task-0")
    agent = DummyAgent()
    record = benchmark.run(agent, task)
    assert record.status == "success"
    assert record.agent_output == "Answer 0"
    assert record.error is None
    assert record.task_id == "mock-task-0"
    assert record.seed == 42


def test_run_error():
    benchmark = MockBenchmark()
    benchmark.load()
    task = benchmark.get_task("mock-task-0")
    agent = FailingAgent()
    record = benchmark.run(agent, task)
    assert record.status == "error"
    assert record.error == "Agent failed"


def test_evaluate():
    benchmark = MockBenchmark()
    benchmark.load()
    task = benchmark.get_task("mock-task-0")

    agent = DummyAgent()
    execution = benchmark.run(agent, task)

    evaluation = benchmark.evaluate(execution)
    assert evaluation.success is True
    assert evaluation.score == 1.0


def test_metadata():
    benchmark = MockBenchmark()
    benchmark.load()
    meta = benchmark.metadata()
    assert meta["name"] == "MockBenchmark"
    assert meta["deterministic"] is True


def test_determinism():
    benchmark1 = MockBenchmark(seed=123)
    benchmark1.load()

    benchmark2 = MockBenchmark(seed=123)
    benchmark2.load()

    agent = DummyAgent()

    task1 = benchmark1.get_task("mock-task-1")
    task2 = benchmark2.get_task("mock-task-1")

    exec1 = benchmark1.run(agent, task1)
    exec2 = benchmark2.run(agent, task2)

    # Execution records should be identical
    assert exec1.sha256() == exec2.sha256()

    eval1 = benchmark1.evaluate(exec1)
    eval2 = benchmark2.evaluate(exec2)

    assert eval1.sha256() == eval2.sha256()


def test_empty_benchmark():
    import llm_reliability.benchmarks.mock_benchmark as mb

    original_tasks = mb.MOCK_TASKS
    mb.MOCK_TASKS = []
    try:
        benchmark = mb.MockBenchmark()
        benchmark.load()
        assert len(benchmark.list_tasks()) == 0
    finally:
        mb.MOCK_TASKS = original_tasks
