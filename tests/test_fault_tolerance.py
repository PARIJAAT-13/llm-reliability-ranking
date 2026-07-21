"""Tests for compute_fault_tolerance."""

import pytest

from llm_reliability.metrics.fault_tolerance import compute_fault_tolerance
from tests.metrics_helpers import make_eval


def test_fault_tolerance_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_fault_tolerance([])


def test_fault_tolerance_no_faulted_raises():
    evs = [make_eval("t1", success=True, score=1.0)]
    with pytest.raises(ValueError, match="No fault-injected"):
        compute_fault_tolerance(evs)


def test_fault_tolerance_perfect():
    # normal 1.0, faulted 1.0 → 1.0
    evs = [
        make_eval("t1", success=True, score=1.0),
        make_eval("t1", success=True, score=1.0, fault_injected=True),
    ]
    assert compute_fault_tolerance(evs) == 1.0


def test_fault_tolerance_zero():
    # normal 1.0, faulted 0.0 → 0.0
    evs = [
        make_eval("t1", success=True, score=1.0),
        make_eval("t1", success=False, score=0.0, fault_injected=True),
    ]
    assert compute_fault_tolerance(evs) == 0.0


def test_fault_tolerance_partial():
    # normal 1.0, faulted 0.75 → 0.75
    evs = [
        make_eval("t1", success=True, score=1.0),
        make_eval("t1", success=True, score=0.75, fault_injected=True),
    ]
    result = compute_fault_tolerance(evs)
    assert abs(result - 0.75) < 1e-9


def test_fault_tolerance_zero_baseline_zero_faulted():
    evs = [
        make_eval("t1", success=False, score=0.0),
        make_eval("t1", success=False, score=0.0, fault_injected=True),
    ]
    assert compute_fault_tolerance(evs) == 0.0


def test_fault_tolerance_in_range():
    evs = [
        make_eval("t1", score=0.9),
        make_eval("t1", score=0.3, fault_injected=True),
    ]
    result = compute_fault_tolerance(evs)
    assert 0.0 <= result <= 1.0
