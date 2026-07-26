"""Tests for WebArena adapter."""

import json

import pytest

from llm_reliability.benchmarks.adapters.webarena_adapter import \
    WebArenaAdapter
from llm_reliability.benchmarks.adapters.webarena_utils import \
    normalize_webarena_answer
from llm_reliability.configs.config import Configuration
from tests.contracts.test_agent_contract import ValidAgent


@pytest.fixture
def sample_dataset(tmp_path):
    path = tmp_path / "sample.json"
    data = [
        {
            "task_id": "w1",
            "prompt": "Find the price of a wireless mouse.",
            "expected_answer": "$29.99",
            "difficulty": "easy",
            "task_category": "web_navigation",
            "metadata": {},
        }
    ]
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def config(sample_dataset):
    return Configuration(
        experiment_name="test",
        benchmark="WebArena",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(sample_dataset)},
    )


def test_webarena_adapter_loading_and_retrieval(config):
    adapter = WebArenaAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert tasks == ["w1"]

    task = adapter.get_task("w1")
    assert task["task_id"] == "w1"
    assert task["prompt"] == "Find the price of a wireless mouse."


def test_webarena_execution_and_evaluation(config):
    adapter = WebArenaAdapter(config)
    adapter.load()
    task = adapter.get_task("w1")

    class SuccessAgent(ValidAgent):
        def run(self, task):
            return "$29.99"

    agent = SuccessAgent()
    execution = adapter.run(agent, task)
    assert execution.status == "success"
    assert execution.agent_output == "$29.99"

    evaluation = adapter.evaluate(execution)
    assert evaluation.success is True
    assert evaluation.score == 1.0


def test_webarena_evaluates_error_as_failure(config):
    adapter = WebArenaAdapter(config)
    adapter.load()
    task = adapter.get_task("w1")

    class ErrorAgent(ValidAgent):
        def run(self, task):
            raise RuntimeError("network error")

    execution = adapter.run(ErrorAgent(), task)
    assert execution.status == "error"

    evaluation = adapter.evaluate(execution)
    assert evaluation.success is False
    assert evaluation.score == 0.0


def test_webarena_utils_normalization():
    assert normalize_webarena_answer("  hello  ") == "hello"
    assert normalize_webarena_answer("Alice!") == "alice"
    assert normalize_webarena_answer("$29.99") == "29.99"
    assert normalize_webarena_answer(None) == ""


def test_webarena_invalid_dataset_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"missing_required": True}]))

    config = Configuration(
        experiment_name="test",
        benchmark="WebArena",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(path)},
    )

    adapter = WebArenaAdapter(config)
    with pytest.raises(ValueError, match="Invalid schema"):
        adapter.load()


def test_webarena_duplicate_ids(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text(
        json.dumps(
            [
                {
                    "task_id": "t1",
                    "prompt": "Task A.",
                    "expected_answer": "A",
                    "difficulty": "easy",
                    "task_category": "cat",
                },
                {
                    "task_id": "t1",
                    "prompt": "Task B.",
                    "expected_answer": "B",
                    "difficulty": "hard",
                    "task_category": "cat",
                },
            ]
        )
    )

    config = Configuration(
        experiment_name="test",
        benchmark="WebArena",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(path)},
    )

    adapter = WebArenaAdapter(config)
    with pytest.raises(ValueError, match="Duplicate task ID"):
        adapter.load()


def test_webarena_metadata_and_determinism(config):
    adapter = WebArenaAdapter(config)
    adapter.load()

    meta = adapter.metadata()
    assert meta["name"] == "WebArena"
    assert meta["task_count"] == 1
    assert meta["deterministic"] is True

    task = adapter.get_task("w1")

    class ErrAgent(ValidAgent):
        def run(self, task):
            raise RuntimeError("fail")

    e1 = adapter.run(ErrAgent(), task)
    e2 = adapter.run(ErrAgent(), task)
    assert e1.sha256() == e2.sha256()


def test_webarena_missing_dataset(tmp_path):
    path = tmp_path / "missing.json"
    config = Configuration(
        experiment_name="test",
        benchmark="WebArena",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(path)},
    )

    adapter = WebArenaAdapter(config)
    with pytest.raises(RuntimeError, match="not found"):
        adapter.load()
