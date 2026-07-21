"""Tests for SWE-bench Lite adapter."""

import json

import pytest

from llm_reliability.benchmarks.adapters.swebench_lite_adapter import SWEBenchLiteAdapter
from llm_reliability.benchmarks.adapters.swebench_utils import normalize_patch
from llm_reliability.configs.config import Configuration
from tests.contracts.test_agent_contract import ValidAgent


@pytest.fixture
def sample_dataset(tmp_path):
    path = tmp_path / "sample.json"
    data = [
        {
            "task_id": "swe_1",
            "repository": "repo",
            "problem_statement": "prob",
            "patch": "patch",
            "expected_outcome": "patch",
            "difficulty": "hard",
            "metadata": {},
        }
    ]
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def config(sample_dataset):
    return Configuration(
        experiment_name="test",
        benchmark="SWEBenchLite",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(sample_dataset)},
    )


def test_swebench_adapter_loading_and_retrieval(config):
    adapter = SWEBenchLiteAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert tasks == ["swe_1"]

    task = adapter.get_task("swe_1")
    assert task["task_id"] == "swe_1"
    assert task["repository"] == "repo"


def test_swebench_execution_and_evaluation(config):
    adapter = SWEBenchLiteAdapter(config)
    adapter.load()
    task = adapter.get_task("swe_1")

    class SuccessAgent(ValidAgent):
        def run(self, task):
            return "patch \n"  # Will test normalization

    agent = SuccessAgent()
    execution = adapter.run(agent, task)
    assert execution.status == "success"
    assert execution.agent_output == "patch \n"

    evaluation = adapter.evaluate(execution)
    assert evaluation.success is True
    assert evaluation.score == 1.0


def test_swebench_utils_normalization():
    assert normalize_patch("patch \n ") == "patch"
    assert normalize_patch("patch\n\n") == "patch"
    assert normalize_patch(None) == ""  # type: ignore


def test_swebench_invalid_dataset_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"missing_required": True}]))

    config = Configuration(
        experiment_name="test",
        benchmark="SWEBenchLite",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(path)},
    )

    adapter = SWEBenchLiteAdapter(config)
    with pytest.raises(ValueError, match="Invalid schema"):
        adapter.load()


def test_swebench_duplicate_ids(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text(
        json.dumps(
            [
                {
                    "task_id": "swe_1",
                    "repository": "r",
                    "problem_statement": "p",
                    "patch": "p",
                    "expected_outcome": "p",
                    "difficulty": "d",
                },
                {
                    "task_id": "swe_1",
                    "repository": "r",
                    "problem_statement": "p",
                    "patch": "p",
                    "expected_outcome": "p",
                    "difficulty": "d",
                },
            ]
        )
    )

    config = Configuration(
        experiment_name="test",
        benchmark="SWEBenchLite",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(path)},
    )

    adapter = SWEBenchLiteAdapter(config)
    with pytest.raises(ValueError, match="Duplicate task ID"):
        adapter.load()


def test_swebench_metadata_and_determinism(config):
    adapter = SWEBenchLiteAdapter(config)
    adapter.load()

    meta = adapter.metadata()
    assert meta["name"] == "SWE-bench Lite"
    assert meta["task_count"] == 1
    assert meta["deterministic"] is True

    task = adapter.get_task("swe_1")

    class ErrAgent(ValidAgent):
        def run(self, task):
            raise RuntimeError("fail")

    e1 = adapter.run(ErrAgent(), task)
    e2 = adapter.run(ErrAgent(), task)
    assert e1.sha256() == e2.sha256()


def test_swebench_missing_dataset(tmp_path):
    path = tmp_path / "missing.json"
    config = Configuration(
        experiment_name="test",
        benchmark="SWEBenchLite",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(path)},
    )

    adapter = SWEBenchLiteAdapter(config)
    with pytest.raises(RuntimeError, match="Missing or invalid dataset"):
        adapter.load()
