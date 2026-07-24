"""Tests for BaseBenchmarkAdapter."""

import pytest

from llm_reliability.benchmarks.adapters.base_adapter import BaseBenchmarkAdapter
from llm_reliability.configs.config import Configuration
from llm_reliability.interfaces.agent import Agent
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord


class DummyAdapter(BaseBenchmarkAdapter):
    """A minimal concrete implementation for testing."""

    def _load_tasks(self) -> None:
        self._tasks = {"task_1": {"task_id": "task_1", "data": 42}}

    def run(self, agent: Agent, task: dict) -> ExecutionRecord:
        return None  # type: ignore

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        return None  # type: ignore


@pytest.fixture
def valid_config():
    return Configuration(
        experiment_name="test",
        benchmark="mock",
        agent="dummy",
        llm="test-llm",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
    )


def test_adapter_inheritance(valid_config):
    adapter = DummyAdapter(config=valid_config)
    assert isinstance(adapter, BaseBenchmarkAdapter)


def test_adapter_validate_configuration():
    with pytest.raises(ValueError, match="Configuration must be provided."):
        DummyAdapter(config=None)  # type: ignore


def test_adapter_validate_dataset(valid_config):
    class EmptyAdapter(BaseBenchmarkAdapter):
        def _load_tasks(self) -> None:
            self._tasks = {}

        def run(self, agent, task):
            pass

        def evaluate(self, execution):
            pass

    adapter = EmptyAdapter(config=valid_config)
    with pytest.raises(ValueError, match="Dataset is empty. No tasks were loaded."):
        adapter.load()


def test_adapter_deterministic_behavior_and_lifecycle(valid_config):
    adapter = DummyAdapter(config=valid_config)

    # Must fail to get tasks before loading
    with pytest.raises(RuntimeError):
        adapter.list_tasks()

    adapter.load()

    tasks = adapter.list_tasks()
    assert tasks == ["task_1"]  # Sorted array returned deterministically

    task = adapter.get_task("task_1")
    assert task["data"] == 42

    meta = adapter.metadata()
    assert meta["deterministic"] is True
    assert meta["task_count"] == 1

    logs = adapter.collect_logs()
    assert len(logs["logs"]) == 1
    assert logs["logs"][0]["event"] == "load"
