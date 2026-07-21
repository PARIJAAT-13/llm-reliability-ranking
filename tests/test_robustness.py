"""Tests for compute_robustness."""

import pytest

from llm_reliability.metrics.robustness import compute_robustness
from tests.metrics_helpers import make_eval


def test_robustness_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_robustness([])


def test_robustness_no_perturbed_raises():
    evs = [make_eval("t1", success=True, score=1.0)]
    with pytest.raises(ValueError, match="No perturbed"):
        compute_robustness(evs)


def test_robustness_perfect():
    # baseline 1.0, perturbed 1.0 → 1.0
    evs = [
        make_eval("t1", success=True, score=1.0),
        make_eval("t1", success=True, score=1.0, perturbation="noise"),
    ]
    assert compute_robustness(evs) == 1.0


def test_robustness_zero_perturbed():
    # baseline 1.0, perturbed 0.0 → 0.0
    evs = [
        make_eval("t1", success=True, score=1.0),
        make_eval("t1", success=False, score=0.0, perturbation="noise"),
    ]
    assert compute_robustness(evs) == 0.0


def test_robustness_partial():
    # baseline 1.0, perturbed 0.5 → 0.5
    evs = [
        make_eval("t1", success=True, score=1.0),
        make_eval("t1", success=True, score=0.5, perturbation="rephrase"),
    ]
    result = compute_robustness(evs)
    assert abs(result - 0.5) < 1e-9


def test_robustness_zero_baseline_zero_perturbed():
    evs = [
        make_eval("t1", success=False, score=0.0),
        make_eval("t1", success=False, score=0.0, perturbation="noise"),
    ]
    assert compute_robustness(evs) == 0.0


def test_robustness_clamped_not_above_1():
    # If somehow perturbed > baseline, clamp to 1.0
    evs = [
        make_eval("t1", success=True, score=0.5),
        make_eval("t1", success=True, score=1.0, perturbation="easy"),
    ]
    result = compute_robustness(evs)
    assert result == 1.0


def test_robustness_in_range():
    evs = [
        make_eval("t1", score=0.8),
        make_eval("t1", score=0.4, perturbation="p"),
    ]
    result = compute_robustness(evs)
    assert 0.0 <= result <= 1.0
