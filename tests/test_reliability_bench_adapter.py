"""Tests for the ReliabilityBench adapter."""

import pytest

from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry
from llm_reliability.benchmarks.adapters.reliability_bench_adapter import (
    RELIABILITY_BENCH_TASKS,
    SCORING_RUBRICS,
    ReliabilityBenchAdapter,
)
from llm_reliability.configs.config import Configuration

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> Configuration:
    return Configuration(
        experiment_name="test",
        benchmark="ReliabilityBench",
        agent="mock",
        llm="gpt-4",
        prompt_version="v1",
        dataset_version="1.0",
        seed=42,
        repetitions=1,
    )


@pytest.fixture
def adapter(config: Configuration) -> ReliabilityBenchAdapter:
    return ReliabilityBenchAdapter(config)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------


def test_built_in_tasks_populated():
    """There must be at least 25 built-in tasks."""
    assert len(RELIABILITY_BENCH_TASKS) >= 25


def test_built_in_tasks_have_required_fields():
    """Every task must have all required metadata fields."""
    required = {
        "task_id",
        "category",
        "domain",
        "difficulty",
        "prompt",
        "reference_answer",
        "scoring_rubric",
        "fault_types",
        "perturbation_types",
    }
    for task in RELIABILITY_BENCH_TASKS:
        missing = required - set(task.keys())
        assert not missing, f"Task {task.get('task_id')} missing: {missing}"


def test_built_in_tasks_categories():
    """All categories must be valid."""
    valid_categories = {
        "reasoning",
        "knowledge",
        "instruction_following",
        "code",
        "language",
        "robustness_probe",
    }
    for task in RELIABILITY_BENCH_TASKS:
        assert (
            task["category"] in valid_categories
        ), f"Task {task['task_id']} has invalid category: {task['category']}"


def test_built_in_tasks_difficulty_range():
    """Difficulty must be in [0, 1]."""
    for task in RELIABILITY_BENCH_TASKS:
        assert (
            0.0 <= task["difficulty"] <= 1.0
        ), f"Task {task['task_id']} difficulty out of range: {task['difficulty']}"


def test_built_in_tasks_scoring_rubrics():
    """All scoring rubrics must be predefined."""
    for task in RELIABILITY_BENCH_TASKS:
        assert (
            task["scoring_rubric"] in SCORING_RUBRICS
        ), f"Task {task['task_id']} has unknown rubric: {task['scoring_rubric']}"


def test_built_in_tasks_unique_ids():
    """All task IDs must be unique."""
    ids = [t["task_id"] for t in RELIABILITY_BENCH_TASKS]
    assert len(ids) == len(set(ids)), "Duplicate task IDs found"


# ---------------------------------------------------------------------------
# Expert tier (difficulty >= 0.9)
# ---------------------------------------------------------------------------

EXPERT_TASK_IDS = {"reason_06", "fact_06", "instruct_06", "code_06", "lang_06", "probe_06"}


def test_expert_tasks_exist():
    """There must be at least 5 expert-tier tasks."""
    expert = [t for t in RELIABILITY_BENCH_TASKS if t["difficulty"] >= 0.9]
    assert len(expert) >= 5, f"Expected >=5 expert tasks, got {len(expert)}"


def test_expert_tasks_have_high_difficulty():
    """Expert-tier tasks must have difficulty >= 0.9 and <= 1.0."""
    for task in RELIABILITY_BENCH_TASKS:
        if task["task_id"] in EXPERT_TASK_IDS:
            assert (
                0.9 <= task["difficulty"] <= 1.0
            ), f"Expert task {task['task_id']} difficulty {task['difficulty']} not in [0.9, 1.0]"


def test_expert_tasks_span_all_categories():
    """Expert tasks should cover at least 4 of the 6 categories."""
    expert_categories = {t["category"] for t in RELIABILITY_BENCH_TASKS if t["difficulty"] >= 0.9}
    assert (
        len(expert_categories) >= 4
    ), f"Expert tasks only cover {len(expert_categories)} categories: {expert_categories}"


def test_expert_tasks_registered():
    """Expert task IDs must be loadable through the adapter."""
    ids = {t["task_id"] for t in RELIABILITY_BENCH_TASKS}
    for eid in EXPERT_TASK_IDS:
        assert eid in ids, f"Expert task {eid} not found in RELIABILITY_BENCH_TASKS"


def test_expert_constraint_match_rubric():
    """Expert constraint match rubric should be in SCORING_RUBRICS."""
    assert "expert_constraint_match" in SCORING_RUBRICS


def test_adapter_evaluate_expert_constraint_partial(adapter: ReliabilityBenchAdapter):
    """Expert constraint rubric should give partial credit."""
    adapter.load()
    agent = MockAgent(
        "Title: The paradox of choice\nDefinition: A concept in psychology.\nExtra text"
    )
    task = adapter.get_task("instruct_06")
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    # Has Title and Definition but missing Example and Citation -> 2/4 = 0.5
    assert not eval_record.success
    assert eval_record.score == 0.5


def test_adapter_evaluate_expert_constraint_full(adapter: ReliabilityBenchAdapter):
    """Expert constraint rubric should give full credit with all sections."""
    adapter.load()
    agent = MockAgent(
        "Title: The paradox of choice\n"
        "Definition: A concept in psychology.\n"
        "Example: Too many options reduce satisfaction.\n"
        "Citation: [Schwartz, 2004]"
    )
    task = adapter.get_task("instruct_06")
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    assert eval_record.success
    assert eval_record.score == 1.0


def test_adapter_evaluate_expert_constraint_missing_title(adapter: ReliabilityBenchAdapter):
    """Expert constraint with no Title should score 0."""
    adapter.load()
    agent = MockAgent("No title here at all")
    task = adapter.get_task("instruct_06")
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    assert not eval_record.success
    assert eval_record.score == 0.0


def test_total_task_count_increased():
    """Total task count should be at least 35 with Expert tier."""
    assert (
        len(RELIABILITY_BENCH_TASKS) >= 35
    ), f"Expected >=35 tasks, got {len(RELIABILITY_BENCH_TASKS)}"


# ---------------------------------------------------------------------------
# Adapter lifecycle
# ---------------------------------------------------------------------------


def test_adapter_requires_config():
    with pytest.raises(Exception):
        ReliabilityBenchAdapter(None)


def test_adapter_initial_state(adapter: ReliabilityBenchAdapter):
    assert not adapter._loaded
    assert adapter._tasks == {}


def test_adapter_loads_built_in_tasks(adapter: ReliabilityBenchAdapter):
    adapter.load()
    assert adapter._loaded
    assert len(adapter._tasks) == len(RELIABILITY_BENCH_TASKS)


def test_adapter_list_tasks(adapter: ReliabilityBenchAdapter):
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) == len(RELIABILITY_BENCH_TASKS)
    assert all(
        t.startswith(("reason_", "fact_", "instruct_", "code_", "lang_", "probe_")) for t in tasks
    )


def test_adapter_get_task(adapter: ReliabilityBenchAdapter):
    adapter.load()
    task = adapter.get_task("reason_01")
    assert task["task_id"] == "reason_01"
    assert task["category"] == "reasoning"
    assert "prompt" in task


def test_adapter_get_task_unknown(adapter: ReliabilityBenchAdapter):
    adapter.load()
    with pytest.raises(ValueError, match="Unknown"):
        adapter.get_task("nonexistent")


def test_adapter_load_twice(adapter: ReliabilityBenchAdapter):
    adapter.load()
    adapter.load()  # Should not raise


# ---------------------------------------------------------------------------
# Run & Evaluate
# ---------------------------------------------------------------------------


class MockAgent:
    def __init__(self, output: str = "Paris"):
        self._output = output

    def run(self, task: dict) -> str:
        return self._output


def test_adapter_run_success(adapter: ReliabilityBenchAdapter):
    adapter.load()
    agent = MockAgent("Paris")
    task = adapter.get_task("fact_04")
    record = adapter.run(agent, task)
    assert record.task_id == "fact_04"
    assert record.status == "success"
    assert record.benchmark == "ReliabilityBench"


def test_adapter_run_agent_error(adapter: ReliabilityBenchAdapter):
    adapter.load()

    class FailingAgent:
        def run(self, task: dict) -> str:
            raise RuntimeError("Agent crashed")

    agent = FailingAgent()
    task = adapter.get_task("reason_01")
    record = adapter.run(agent, task)
    assert record.status == "error"
    assert record.error is not None


def test_adapter_evaluate_exact_match_success(adapter: ReliabilityBenchAdapter):
    adapter.load()
    agent = MockAgent("yes")
    task = adapter.get_task("reason_01")
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    assert eval_record.success
    assert eval_record.score == 1.0


def test_adapter_evaluate_exact_match_failure(adapter: ReliabilityBenchAdapter):
    adapter.load()
    agent = MockAgent("no")
    task = adapter.get_task("reason_01")
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    assert not eval_record.success
    assert eval_record.score == 0.0


def test_adapter_evaluate_numeric_match(adapter: ReliabilityBenchAdapter):
    adapter.load()
    agent = MockAgent("150")
    task = adapter.get_task("reason_02")
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    assert eval_record.success
    assert eval_record.score == 1.0


def test_adapter_evaluate_contains_match(adapter: ReliabilityBenchAdapter):
    adapter.load()
    agent = MockAgent("The bug is that it uses subtraction instead of addition.")
    task = adapter.get_task("code_02")
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    assert eval_record.success


def test_adapter_evaluate_json_format(adapter: ReliabilityBenchAdapter):
    adapter.load()
    agent = MockAgent('{"answer": 12, "confidence": 0.95}')
    task = adapter.get_task("probe_03")
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    assert eval_record.success
    assert eval_record.score == 1.0


def test_adapter_evaluate_error_execution(adapter: ReliabilityBenchAdapter):
    adapter.load()

    class FailingAgent:
        def run(self, task: dict) -> str:
            raise RuntimeError("fail")

    agent = FailingAgent()
    task = adapter.get_task("fact_01")
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    assert not eval_record.success
    assert eval_record.score == 0.0


def test_adapter_evaluate_multi_step(adapter: ReliabilityBenchAdapter):
    adapter.load()
    agent = MockAgent("Step 1 complete\n42\nStep 3 complete")
    task = adapter.get_task("instruct_03")
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    assert eval_record.success
    assert eval_record.score == 1.0


def test_adapter_evaluate_multi_item(adapter: ReliabilityBenchAdapter):
    adapter.load()
    agent = MockAgent(
        "Interpretation 1: He owns the telescope\nInterpretation 2: He uses the telescope"
    )
    task = adapter.get_task("lang_05")
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    assert eval_record.success


def test_adapter_evaluate_negation_compliance(adapter: ReliabilityBenchAdapter):
    adapter.load()
    agent = MockAgent("I enjoy summer for its warm weather and long days.")
    task = adapter.get_task("instruct_05")
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    assert eval_record.success


def test_adapter_evaluate_negation_violation(adapter: ReliabilityBenchAdapter):
    adapter.load()
    agent = MockAgent("I like the color blue and dogs.")
    task = adapter.get_task("instruct_05")
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    assert not eval_record.success


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_reliability_bench():
    assert BenchmarkRegistry.exists("ReliabilityBench")


def test_registry_get_adapter():
    cls = BenchmarkRegistry.get("ReliabilityBench")
    assert cls is ReliabilityBenchAdapter


# ---------------------------------------------------------------------------
# Metadata & logging
# ---------------------------------------------------------------------------


def test_adapter_metadata(adapter: ReliabilityBenchAdapter):
    adapter.load()
    meta = adapter.metadata()
    assert meta["name"] == "ReliabilityBench"
    assert meta["task_count"] == len(RELIABILITY_BENCH_TASKS)
    assert meta["deterministic"]


def test_adapter_collect_logs(adapter: ReliabilityBenchAdapter):
    adapter.load()
    agent = MockAgent("yes")
    task = adapter.get_task("reason_01")
    adapter.run(agent, task)
    logs = adapter.collect_logs()
    assert len(logs["logs"]) >= 1
