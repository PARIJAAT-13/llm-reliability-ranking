"""Tests for the Prompt Perturbation Framework."""

import pytest
from typing import Any

from llm_reliability.agents.mock_agent import MockAgent
from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
from llm_reliability.configs.config import Configuration
from llm_reliability.metrics.robustness import compute_robustness
from llm_reliability.reliability.perturbation import (
    FormattingPerturbationStrategy,
    InstructionReorderingPerturbationStrategy,
    PerturbationManager,
    PerturbationRunResult,
    PerturbationStrategy,
    PromptWrapperPerturbationStrategy,
    SynonymSubstitutionPerturbationStrategy,
    WhitespacePerturbationStrategy,
)


class FaultyStrategy(PerturbationStrategy):
    """Strategy that intentionally raises an exception to test error handling."""

    @property
    def name(self) -> str:
        return "faulty"

    @property
    def description(self) -> str:
        return "Always raises an Exception."

    def apply(self, task: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
        raise RuntimeError("Intentionally failing strategy.")


@pytest.fixture
def sample_task() -> dict[str, Any]:
    return {
        "task_id": "task-test-1",
        "prompt": "Solve mock problem 1. Calculate the result.",
        "expected_answer": "Answer 1",
        "difficulty": "easy",
        "category": "logic",
    }


def test_whitespace_strategy(sample_task: dict[str, Any]):
    strategy = WhitespacePerturbationStrategy()
    assert strategy.name == "whitespace"
    assert "whitespace" in strategy.description.lower()

    perturbed = strategy.apply(sample_task, seed=42)
    assert perturbed["task_id"] == sample_task["task_id"]
    assert perturbed["prompt"] != sample_task["prompt"]
    assert perturbed["metadata"]["perturbation"]["strategy"] == "whitespace"


def test_formatting_strategy():
    strategy = FormattingPerturbationStrategy()
    assert strategy.name == "formatting"

    # Test numbered to bullet conversion
    numbered_task = {
        "task_id": "t1",
        "prompt": "Instructions:\n1. First step.\n2. Second step.",
    }
    perturbed = strategy.apply(numbered_task, seed=42)
    assert "* First step." in perturbed["prompt"]

    # Test bullet to numbered conversion
    bullet_task = {
        "task_id": "t2",
        "prompt": "Instructions:\n- First item.\n- Second item.",
    }
    perturbed_num = strategy.apply(bullet_task, seed=42)
    assert "1. First item." in perturbed_num["prompt"]


def test_instruction_reordering_strategy():
    strategy = InstructionReorderingPerturbationStrategy()
    assert strategy.name == "reordering"

    # List item reordering
    bullet_task = {
        "task_id": "t3_bullet",
        "prompt": "- Task A\n- Task B\n- Task C\n- Task D",
    }
    perturbed = strategy.apply(bullet_task, seed=42)
    assert perturbed["prompt"] != bullet_task["prompt"]


def test_synonym_substitution_strategy():
    strategy = SynonymSubstitutionPerturbationStrategy()
    assert strategy.name == "synonym"

    task = {
        "task_id": "t4",
        "prompt": "Solve the question and calculate the answer.",
    }
    perturbed = strategy.apply(task, seed=42)
    assert perturbed["prompt"] != task["prompt"]
    # Technical words like numbers and formatting are preserved
    code_task = {
        "task_id": "t4_code",
        "prompt": "def solve(x): return calculate(x)",
    }
    perturbed_code = strategy.apply(code_task, seed=42)
    assert "def" in perturbed_code["prompt"]
    assert "return" in perturbed_code["prompt"]


def test_prompt_wrapper_strategy(sample_task: dict[str, Any]):
    strategy = PromptWrapperPerturbationStrategy()
    assert strategy.name == "prompt_wrapper"

    perturbed = strategy.apply(sample_task, seed=100)
    assert sample_task["prompt"] in perturbed["prompt"]
    assert len(perturbed["prompt"]) > len(sample_task["prompt"])


def test_manager_strategy_selection(sample_task: dict[str, Any]):
    cfg = Configuration(
        experiment_name="test_pert",
        benchmark="MockBenchmark",
        agent="MockAgent",
        llm="mock",
        prompt_version="v1",
        dataset_version="v1",
        seed=42,
        repetitions=1,
        perturbations=("whitespace", "synonym"),
    )
    manager = PerturbationManager(config=cfg)
    assert len(manager.enabled_strategies) == 2
    names = [s.name for s in manager.enabled_strategies]
    assert "whitespace" in names
    assert "synonym" in names

    variants = manager.generate_perturbations(sample_task)
    assert len(variants) == 2


def test_manager_max_perturbations_cap(sample_task: dict[str, Any]):
    manager = PerturbationManager(max_perturbations=2, seed=42)
    variants = manager.generate_perturbations(sample_task)
    assert len(variants) == 2


def test_manager_fault_tolerance(sample_task: dict[str, Any]):
    faulty = FaultyStrategy()
    normal = WhitespacePerturbationStrategy()
    manager = PerturbationManager(strategies=[faulty, normal])

    # Faulty strategy error should be logged and skipped; normal strategy succeeds
    variants = manager.generate_perturbations(sample_task)
    assert len(variants) == 1
    assert variants[0]["metadata"]["perturbation"]["strategy"] == "whitespace"


def test_end_to_end_perturbed_run():
    cfg = Configuration(
        experiment_name="pert_e2e",
        benchmark="MockBenchmark",
        agent="MockAgent",
        llm="mock",
        prompt_version="v1",
        dataset_version="v1",
        seed=42,
        repetitions=1,
        perturbations=("whitespace", "formatting", "prompt_wrapper"),
    )
    benchmark = MockBenchmark(config=cfg)
    benchmark.load()
    agent = MockAgent(config=cfg)
    task = benchmark.get_task("mock-task-0")

    manager = PerturbationManager(config=cfg)
    res: PerturbationRunResult = manager.run_perturbed_task(
        agent=agent,
        benchmark=benchmark,
        task=task,
    )

    assert isinstance(res, PerturbationRunResult)
    assert res.task_id == task["task_id"]
    # 1 baseline + 3 perturbed executions
    assert len(res.execution_records) == 4
    assert len(res.evaluation_records) == 4

    assert res.baseline_execution is not None
    assert res.baseline_execution.perturbation is None

    assert len(res.perturbed_executions) == 3
    for exec_rec in res.perturbed_executions:
        assert exec_rec.perturbation is not None
        assert exec_rec.task_id == task["task_id"]

    # Verify downstream robustness calculation compatibility
    robustness = compute_robustness(res.evaluation_records)
    assert 0.0 <= robustness <= 1.0
