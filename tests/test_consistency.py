"""Tests for compute_consistency."""

import pytest

from llm_reliability.metrics.consistency import compute_consistency
from tests.metrics_helpers import make_eval


def test_consistency_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_consistency([])


def test_consistency_all_success_single_task():
    evs = [make_eval("t1", success=True, score=1.0) for _ in range(5)]
    result = compute_consistency(evs)
    # mean=1.0, std=0.0 → 1.0
    assert result == 1.0


def test_consistency_all_failure_single_task():
    evs = [make_eval("t1", success=False, score=0.0) for _ in range(5)]
    result = compute_consistency(evs)
    # mean=0.0, std=0.0 → 0.0
    assert result == 0.0


def test_consistency_mixed_tasks_uniform():
    # task A: always success, task B: always success → std=0 → cons = mean = 1.0
    evs = [
        make_eval("A", True, 1.0),
        make_eval("A", True, 1.0),
        make_eval("B", True, 1.0),
        make_eval("B", True, 1.0),
    ]
    assert compute_consistency(evs) == 1.0


def test_consistency_mixed_tasks_divergent():
    # task A: 100%, task B: 0% → std = 0.5, mean = 0.5 → 0.0 clamped
    evs = [
        make_eval("A", True, 1.0),
        make_eval("A", True, 1.0),
        make_eval("B", False, 0.0),
        make_eval("B", False, 0.0),
    ]
    result = compute_consistency(evs)
    assert result == 0.0


def test_consistency_partial():
    # task A: 50%, task B: 100% → per-task rates [0.5, 1.0], mean=0.75, std=0.25 → 0.5
    evs = [
        make_eval("A", True, 1.0),
        make_eval("A", False, 0.0),
        make_eval("B", True, 1.0),
        make_eval("B", True, 1.0),
    ]
    result = compute_consistency(evs)
    assert abs(result - 0.5) < 1e-9


def test_consistency_deterministic():
    evs = [make_eval(f"t{i}", success=i % 2 == 0) for i in range(10)]
    assert compute_consistency(evs) == compute_consistency(evs)


def test_consistency_in_range():
    evs = [make_eval(f"t{i}", success=i % 3 == 0) for i in range(9)]
    result = compute_consistency(evs)
    assert 0.0 <= result <= 1.0
