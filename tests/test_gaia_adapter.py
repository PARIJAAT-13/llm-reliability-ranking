"""Tests for GAIA adapter."""

import json

import pytest

from llm_reliability.benchmarks.adapters.gaia_adapter import GAIAAdapter
from llm_reliability.benchmarks.adapters.gaia_utils import \
    normalize_gaia_answer
from llm_reliability.configs.config import Configuration
from tests.contracts.test_agent_contract import ValidAgent


@pytest.fixture
def sample_dataset(tmp_path):
    path = tmp_path / "sample.json"
    data = [
        {
            "task_id": "g1",
            "question": "q",
            "ground_truth_answer": "ans",
            "difficulty": "easy",
            "task_category": "cat",
            "metadata": {},
        }
    ]
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def config(sample_dataset):
    return Configuration(
        experiment_name="test",
        benchmark="GAIA",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(sample_dataset)},
    )


def test_gaia_adapter_loading_and_retrieval(config):
    adapter = GAIAAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert tasks == ["g1"]

    task = adapter.get_task("g1")
    assert task["task_id"] == "g1"
    assert task["question"] == "q"


def test_gaia_execution_and_evaluation(config):
    adapter = GAIAAdapter(config)
    adapter.load()
    task = adapter.get_task("g1")

    class SuccessAgent(ValidAgent):
        def run(self, task):
            return "ANS."  # Will test normalization

    agent = SuccessAgent()
    execution = adapter.run(agent, task)
    assert execution.status == "success"
    assert execution.agent_output == "ANS."

    evaluation = adapter.evaluate(execution)
    assert evaluation.success is True
    assert evaluation.score == 1.0


def test_gaia_utils_normalization():
    assert normalize_gaia_answer("Paris.") == "paris"
    assert normalize_gaia_answer(" Paris ") == "paris"
    assert normalize_gaia_answer("PARIS!") == "paris"
    assert normalize_gaia_answer(None) == ""


def test_gaia_invalid_dataset_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"missing_required": True}]))

    config = Configuration(
        experiment_name="test",
        benchmark="GAIA",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(path)},
    )

    adapter = GAIAAdapter(config)
    with pytest.raises(ValueError, match="Invalid schema"):
        adapter.load()


def test_gaia_duplicate_ids(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text(
        json.dumps(
            [
                {
                    "task_id": "t1",
                    "question": "a",
                    "ground_truth_answer": "a",
                    "difficulty": "a",
                    "task_category": "a",
                },
                {
                    "task_id": "t1",
                    "question": "b",
                    "ground_truth_answer": "b",
                    "difficulty": "b",
                    "task_category": "b",
                },
            ]
        )
    )

    config = Configuration(
        experiment_name="test",
        benchmark="GAIA",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(path)},
    )

    adapter = GAIAAdapter(config)
    with pytest.raises(ValueError, match="Duplicate task ID"):
        adapter.load()


def test_gaia_metadata_and_determinism(config):
    adapter = GAIAAdapter(config)
    adapter.load()

    meta = adapter.metadata()
    assert meta["name"] == "GAIA"
    assert meta["task_count"] == 1
    assert meta["deterministic"] is True

    task = adapter.get_task("g1")

    class ErrAgent(ValidAgent):
        def run(self, task):
            raise RuntimeError("fail")

    e1 = adapter.run(ErrAgent(), task)
    e2 = adapter.run(ErrAgent(), task)
    assert e1.sha256() == e2.sha256()


def test_gaia_missing_dataset(tmp_path):
    path = tmp_path / "missing.json"
    config = Configuration(
        experiment_name="test",
        benchmark="GAIA",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(path)},
    )

    adapter = GAIAAdapter(config)
    with pytest.raises(RuntimeError, match="Missing or invalid dataset"):
        adapter.load()
