"""Tests for the ReliabilityEngine."""

import pytest

from llm_reliability.metrics import ReliabilityEngine, ReliabilityResult
from tests.metrics_helpers import make_eval


def _all_success(n: int = 5, n_tasks: int = 1):
    return [
        make_eval(task_id=f"t{i % n_tasks}", success=True, score=1.0, run_index=i) for i in range(n)
    ]


def _all_failure(n: int = 5, n_tasks: int = 1):
    return [
        make_eval(task_id=f"t{i % n_tasks}", success=False, score=0.0, run_index=i)
        for i in range(n)
    ]


def test_engine_requires_evaluations():
    with pytest.raises(ValueError, match="at least one"):
        ReliabilityEngine([])


def test_compute_all_returns_reliability_result():
    engine = ReliabilityEngine(_all_success(10, n_tasks=2))
    result = engine.compute_all()
    assert isinstance(result, ReliabilityResult)


def test_compute_all_perfect_agent():
    engine = ReliabilityEngine(_all_success(10, n_tasks=2))
    result = engine.compute_all()
    assert result.success_rate == 1.0
    assert result.consistency == 1.0
    assert result.composite == 1.0
    assert result.n_evaluations == 10


def test_compute_all_zero_agent():
    engine = ReliabilityEngine(_all_failure(10, n_tasks=2))
    result = engine.compute_all()
    assert result.success_rate == 0.0
    # consistency uses std of per-task rates: all 0 → std=0, mean=0 → 0
    assert result.consistency == 0.0
    assert result.composite == 0.0


def test_compute_all_with_task_id():
    engine = ReliabilityEngine(_all_success(4))
    result = engine.compute_all(task_id="my_task")
    assert result.task_id == "my_task"


def test_compute_all_custom_weights():
    evs = _all_success(4, n_tasks=2)
    engine = ReliabilityEngine(evs)
    result = engine.compute_all(weights={"success_rate": 0.6, "consistency": 0.4})
    assert abs(result.composite - 1.0) < 1e-6
    assert result.weights == {"success_rate": 0.6, "consistency": 0.4}


def test_compute_all_with_perturbations():
    baseline = [make_eval("t1", success=True, score=1.0)]
    perturbed = [make_eval("t1", success=True, score=0.8, perturbation="noise")]
    engine = ReliabilityEngine(baseline + perturbed)
    result = engine.compute_all()
    assert result.robustness is not None
    assert 0.0 <= result.robustness <= 1.0


def test_compute_all_without_perturbations():
    engine = ReliabilityEngine(_all_success(4))
    result = engine.compute_all()
    assert result.robustness is None


def test_compute_all_with_fault_injection():
    normal = [make_eval("t1", success=True, score=1.0)]
    faulted = [make_eval("t1", success=False, score=0.0, fault_injected=True)]
    engine = ReliabilityEngine(normal + faulted)
    result = engine.compute_all()
    assert result.fault_tolerance is not None
    assert result.fault_tolerance == 0.0


def test_deterministic_results():
    evs = [make_eval(f"t{i}", success=i % 2 == 0, score=float(i % 2)) for i in range(10)]
    r1 = ReliabilityEngine(evs).compute_all()
    r2 = ReliabilityEngine(evs).compute_all()
    assert r1.success_rate == r2.success_rate
    assert r1.consistency == r2.consistency
    assert r1.composite == r2.composite
