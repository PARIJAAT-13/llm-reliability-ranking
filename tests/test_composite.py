"""Tests for compute_composite."""

import pytest

from llm_reliability.metrics.composite import compute_composite


def test_composite_equal_weights_two_metrics():
    composite, weights = compute_composite(success_rate=1.0, consistency=1.0)
    assert composite == 1.0
    assert abs(weights["success_rate"] - 0.5) < 1e-9
    assert abs(weights["consistency"] - 0.5) < 1e-9


def test_composite_equal_weights_four_metrics():
    composite, weights = compute_composite(
        success_rate=1.0, consistency=1.0, robustness=1.0, fault_tolerance=1.0
    )
    assert composite == 1.0
    for v in weights.values():
        assert abs(v - 0.25) < 1e-9


def test_composite_zero():
    composite, _ = compute_composite(success_rate=0.0, consistency=0.0)
    assert composite == 0.0


def test_composite_partial():
    composite, _ = compute_composite(success_rate=0.8, consistency=0.6)
    assert abs(composite - 0.7) < 1e-9


def test_composite_custom_weights():
    composite, weights = compute_composite(
        success_rate=1.0,
        consistency=0.0,
        weights={"success_rate": 0.8, "consistency": 0.2},
    )
    assert abs(composite - 0.8) < 1e-9


def test_composite_normalizes_weights():
    composite, weights = compute_composite(
        success_rate=1.0,
        consistency=0.5,
        weights={"success_rate": 2.0, "consistency": 1.0},
    )
    assert abs(weights["success_rate"] - 2.0 / 3.0) < 1e-9
    assert abs(weights["consistency"] - 1.0 / 3.0) < 1e-9
    expected = (2.0 / 3.0) * 1.0 + (1.0 / 3.0) * 0.5
    assert abs(composite - expected) < 1e-9


def test_composite_normalizes_uneven_sum():
    composite, weights = compute_composite(
        success_rate=1.0,
        consistency=1.0,
        weights={"success_rate": 3.0, "consistency": 1.0},
    )
    assert abs(weights["success_rate"] - 0.75) < 1e-9
    assert abs(weights["consistency"] - 0.25) < 1e-9


def test_composite_rejects_zero_total_weights():
    with pytest.raises(ValueError, match="positive value"):
        compute_composite(
            success_rate=1.0,
            consistency=1.0,
            weights={"success_rate": 0.0, "consistency": 0.0},
        )


def test_composite_rejects_negative_weight():
    with pytest.raises(ValueError, match="cannot be negative"):
        compute_composite(
            success_rate=1.0,
            consistency=1.0,
            weights={"success_rate": -0.5, "consistency": 1.5},
        )


def test_composite_weights_unknown_key():
    with pytest.raises(ValueError, match="Unknown metric keys"):
        compute_composite(
            success_rate=1.0,
            consistency=1.0,
            weights={"success_rate": 0.5, "unknown": 0.5},
        )


def test_composite_weights_missing_metric_key():
    with pytest.raises(ValueError, match="missing"):
        compute_composite(
            success_rate=1.0,
            consistency=1.0,
            weights={"success_rate": 1.0},
        )


def test_composite_with_optional_metrics():
    composite, weights = compute_composite(
        success_rate=0.8,
        consistency=0.6,
        robustness=1.0,
        fault_tolerance=None,
    )
    assert "robustness" in weights
    assert "fault_tolerance" not in weights
    assert abs(composite - 0.8) < 1e-6


def test_composite_clamped_to_range():
    composite, _ = compute_composite(success_rate=1.0, consistency=1.0)
    assert 0.0 <= composite <= 1.0


def test_composite_equal_weights_with_all_none_optionals():
    composite, weights = compute_composite(
        success_rate=1.0, consistency=0.0, robustness=None, fault_tolerance=None
    )
    assert weights.keys() == {"success_rate", "consistency"}
