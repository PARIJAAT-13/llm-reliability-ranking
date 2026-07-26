"""Integration tests for full GAIA evaluation support."""

import json

import pytest

from llm_reliability.benchmarks.adapters.gaia_adapter import GAIAAdapter
from llm_reliability.configs.config import Configuration
from tests.contracts.test_agent_contract import ValidAgent


@pytest.fixture
def multi_level_dataset(tmp_path):
    """GAIA dataset with tasks across Levels 1, 2, 3."""
    path = tmp_path / "gaia_multi.json"
    data = [
        {
            "task_id": "g_l1_001",
            "question": "Capital of France?",
            "ground_truth_answer": "Paris",
            "difficulty": "1",
            "task_category": "geography",
            "metadata": {},
        },
        {
            "task_id": "g_l2_001",
            "question": "Derive the quadratic formula.",
            "ground_truth_answer": "x = (-b ± sqrt(b²-4ac))/2a",
            "difficulty": "2",
            "task_category": "math",
            "metadata": {},
        },
        {
            "task_id": "g_l3_001",
            "question": "Prove the Riemann Hypothesis.",
            "ground_truth_answer": "unsolved",
            "difficulty": "3",
            "task_category": "math",
            "metadata": {},
        },
        {
            "task_id": "g_l1_002",
            "question": "What is 2+2?",
            "ground_truth_answer": "4",
            "difficulty": "1",
            "task_category": "math",
            "metadata": {},
        },
    ]
    path.write_text(json.dumps(data))
    return path


def test_gaia_loads_all_levels_by_default(multi_level_dataset):
    config = Configuration(
        experiment_name="test",
        benchmark="GAIA",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(multi_level_dataset)},
    )
    adapter = GAIAAdapter(config)
    adapter.load()
    assert len(adapter.list_tasks()) == 4


def test_gaia_filter_level_1_only(multi_level_dataset):
    config = Configuration(
        experiment_name="test",
        benchmark="GAIA",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(multi_level_dataset), "levels": [1]},
    )
    adapter = GAIAAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) == 2
    assert "g_l1_001" in tasks
    assert "g_l1_002" in tasks


def test_gaia_filter_levels_2_and_3(multi_level_dataset):
    config = Configuration(
        experiment_name="test",
        benchmark="GAIA",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(multi_level_dataset), "levels": [2, 3]},
    )
    adapter = GAIAAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) == 2
    assert "g_l2_001" in tasks
    assert "g_l3_001" in tasks


def test_gaia_system_prompt_injected(multi_level_dataset):
    config = Configuration(
        experiment_name="test",
        benchmark="GAIA",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(multi_level_dataset)},
    )
    adapter = GAIAAdapter(config)
    adapter.load()
    task = adapter.get_task("g_l1_001")

    class RecordAgent(ValidAgent):
        def __init__(self):
            self.received = None

        def run(self, task):
            self.received = task
            return "Paris"

    agent = RecordAgent()
    adapter.run(agent, task)
    assert agent.received is not None
    assert "system_prompt" in agent.received
    assert "GAIA benchmark" in agent.received["system_prompt"]
    assert "Return ONLY" in agent.received["system_prompt"]


def test_gaia_evaluates_with_system_prompt():
    """System prompt does not affect evaluation scoring."""
    import hashlib
    import json as _json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/single.json"
        with open(path, "w") as f:
            _json.dump(
                [
                    {
                        "task_id": "t1",
                        "question": "Q?",
                        "ground_truth_answer": "Answer",
                        "difficulty": "1",
                        "task_category": "gen",
                    }
                ],
                f,
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
            metadata={"dataset_path": path},
        )

        adapter = GAIAAdapter(config)
        adapter.load()
        task = adapter.get_task("t1")

        class GoodAgent(ValidAgent):
            def run(self, task):
                return "Answer"

        execution = adapter.run(GoodAgent(), task)
        evaluation = adapter.evaluate(execution)
        assert evaluation.success is True
        assert evaluation.score == 1.0

        class WrongAgent(ValidAgent):
            def run(self, task):
                return "Wrong"

        exec2 = adapter.run(WrongAgent(), task)
        eval2 = adapter.evaluate(exec2)
        assert eval2.success is False
        assert eval2.score == 0.0


def test_gaia_multiple_seeds_deterministic():
    """Same seed produces same timing across runs."""
    import hashlib
    import json as _json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/det.json"
        with open(path, "w") as f:
            _json.dump(
                [
                    {
                        "task_id": "d1",
                        "question": "Q?",
                        "ground_truth_answer": "A",
                        "difficulty": "1",
                        "task_category": "gen",
                    }
                ],
                f,
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
            metadata={"dataset_path": path},
        )

        adapter = GAIAAdapter(config)
        adapter.load()
        task = adapter.get_task("d1")

        class MyAgent(ValidAgent):
            def run(self, task):
                return "A"

        e1 = adapter.run(MyAgent(), task)
        e2 = adapter.run(MyAgent(), task)
        assert e1.runtime_seconds == e2.runtime_seconds
        assert e1.sha256() == e2.sha256()


def test_gaia_evaluation_with_empty_output(multi_level_dataset):
    config = Configuration(
        experiment_name="test",
        benchmark="GAIA",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": str(multi_level_dataset)},
    )
    adapter = GAIAAdapter(config)
    adapter.load()
    task = adapter.get_task("g_l1_001")

    class EmptyAgent(ValidAgent):
        def run(self, task):
            return ""

    execution = adapter.run(EmptyAgent(), task)
    evaluation = adapter.evaluate(execution)
    assert evaluation.success is False
    assert evaluation.score == 0.0
