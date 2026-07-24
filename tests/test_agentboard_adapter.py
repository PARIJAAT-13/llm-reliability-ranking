"""Tests for AgentBoard adapter."""

import json

import pytest

from llm_reliability.benchmarks.adapters.agentboard_adapter import AgentBoardAdapter
from llm_reliability.configs.config import Configuration
from tests.contracts.test_agent_contract import ValidAgent


@pytest.fixture
def sample_dataset(tmp_path):
    path = tmp_path / "sample.json"
    data = [
        {
            "task_id": "t1",
            "prompt": "do something",
            "expected_output": "success",
            "difficulty": "easy",
            "category": "web",
            "metadata": {},
        }
    ]
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def config(sample_dataset):
    return Configuration(
        experiment_name="test",
        benchmark="AgentBoard",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(sample_dataset)},
    )


def test_agentboard_adapter_loading_and_retrieval(config):
    adapter = AgentBoardAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert tasks == ["t1"]

    task = adapter.get_task("t1")
    assert task["task_id"] == "t1"
    assert task["prompt"] == "do something"


def test_agentboard_execution_and_evaluation(config):
    adapter = AgentBoardAdapter(config)
    adapter.load()
    task = adapter.get_task("t1")

    class SuccessAgent(ValidAgent):
        def run(self, task):
            return "success"

    agent = SuccessAgent()
    execution = adapter.run(agent, task)
    assert execution.status == "success"
    assert execution.agent_output == "success"

    evaluation = adapter.evaluate(execution)
    assert evaluation.success is True
    assert evaluation.score == 1.0


def test_agentboard_invalid_dataset_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"missing_required": True}]))

    config = Configuration(
        experiment_name="test",
        benchmark="AgentBoard",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(path)},
    )

    adapter = AgentBoardAdapter(config)
    with pytest.raises(ValueError, match="Invalid schema"):
        adapter.load()


def test_agentboard_duplicate_ids(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text(
        json.dumps(
            [
                {
                    "task_id": "t1",
                    "prompt": "a",
                    "expected_output": "a",
                    "difficulty": "a",
                    "category": "a",
                },
                {
                    "task_id": "t1",
                    "prompt": "b",
                    "expected_output": "b",
                    "difficulty": "b",
                    "category": "b",
                },
            ]
        )
    )

    config = Configuration(
        experiment_name="test",
        benchmark="AgentBoard",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(path)},
    )

    adapter = AgentBoardAdapter(config)
    with pytest.raises(ValueError, match="Duplicate task ID"):
        adapter.load()


def test_agentboard_metadata_and_determinism(config):
    adapter = AgentBoardAdapter(config)
    adapter.load()

    meta = adapter.metadata()
    assert meta["name"] == "AgentBoard"
    assert meta["task_count"] == 1
    assert meta["deterministic"] is True

    task = adapter.get_task("t1")

    class ErrAgent(ValidAgent):
        def run(self, task):
            raise RuntimeError("fail")

    e1 = adapter.run(ErrAgent(), task)
    e2 = adapter.run(ErrAgent(), task)
    assert e1.sha256() == e2.sha256()


def test_agentboard_configuration_validation():
    config = Configuration(
        experiment_name="test",
        benchmark="AgentBoard",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={},
    )
    with pytest.raises(ValueError, match="must contain 'dataset_path'"):
        AgentBoardAdapter(config)


def test_agentboard_missing_dataset(tmp_path):
    path = tmp_path / "missing.json"
    config = Configuration(
        experiment_name="test",
        benchmark="AgentBoard",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(path)},
    )

    adapter = AgentBoardAdapter(config)
    with pytest.raises(RuntimeError, match="Missing or invalid dataset"):
        adapter.load()
