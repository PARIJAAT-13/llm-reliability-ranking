"""Tests for model calibration metrics (ECE, MCE, Brier score, reliability diagrams)."""

from __future__ import annotations

import pytest

from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.reliability.metrics.calibration import (
    CalibrationResult,
    _reliability_diagram_data,
    compute_brier_score,
    compute_calibration,
    compute_ece,
    compute_mce,
)

# Helpers -------------------------------------------------------------------


def _make_eval(score: float, success: bool) -> EvaluationRecord:
    return EvaluationRecord(
        execution_hash="a" * 64,
        configuration_hash="b" * 64,
        seed=0,
        benchmark="test",
        agent="test_agent",
        task_id="t1",
        run_index=0,
        success=success,
        score=score,
        evaluated_at="2025-01-01T00:00:00",
    )


# compute_ece ---------------------------------------------------------------


def test_ece_perfect_calibration() -> None:
    confidences = [0.5, 0.5, 0.5, 0.5]
    outcomes = [True, False, True, False]
    ece, bin_accs, bin_confs, bin_counts = compute_ece(confidences, outcomes, n_bins=5)
    assert ece == 0.0


def test_ece_imperfect_calibration() -> None:
    confidences = [0.9, 0.9, 0.9, 0.9, 0.9]
    outcomes = [True, True, True, True, False]
    ece, bin_accs, bin_confs, bin_counts = compute_ece(confidences, outcomes, n_bins=10)
    assert ece > 0.0
    assert ece < 1.0


def test_ece_singlebin() -> None:
    confidences = [0.9, 0.8, 0.7]
    outcomes = [True, True, False]
    ece, bin_accs, bin_confs, bin_counts = compute_ece(confidences, outcomes, n_bins=1)
    bin_acc = sum(outcomes) / len(outcomes)
    bin_conf = sum(confidences) / len(confidences)
    expected = abs(bin_acc - bin_conf)
    assert ece == pytest.approx(expected)


def test_ece_all_wrong() -> None:
    confidences = [0.9, 0.8, 0.7]
    outcomes = [False, False, False]
    ece, bin_accs, bin_confs, bin_counts = compute_ece(confidences, outcomes, n_bins=3)
    assert ece > 0.0


def test_ece_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_ece([], [])


def test_ece_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_ece([0.5], [True, False])


def test_ece_boundary_values() -> None:
    confidences = [0.0, 0.5, 1.0]
    outcomes = [False, True, True]
    ece, bin_accs, bin_confs, bin_counts = compute_ece(confidences, outcomes, n_bins=3)
    assert 0.0 <= ece <= 1.0


# compute_mce ---------------------------------------------------------------


def test_mce_same_as_ece_for_one_bin() -> None:
    confidences = [0.9, 0.8, 0.7]
    outcomes = [True, True, False]
    mce = compute_mce(confidences, outcomes, n_bins=1)
    ece, _, _, _ = compute_ece(confidences, outcomes, n_bins=1)
    assert mce == ece


def test_mce_greater_or_equal_to_ece() -> None:
    confidences = [0.5, 0.6, 0.7, 0.8, 0.9]
    outcomes = [True, False, True, False, True]
    ece, _, _, _ = compute_ece(confidences, outcomes, n_bins=3)
    mce = compute_mce(confidences, outcomes, n_bins=3)
    assert mce >= ece


def test_mce_empty_bins() -> None:
    confidences = [0.01, 0.02, 0.99, 0.99]
    outcomes = [True, False, True, True]
    mce = compute_mce(confidences, outcomes, n_bins=10)
    assert isinstance(mce, float)
    assert 0.0 <= mce <= 1.0


# compute_brier_score -------------------------------------------------------


def test_brier_perfect() -> None:
    assert compute_brier_score([1.0, 0.0], [True, False]) == 0.0


def test_brier_worst() -> None:
    assert compute_brier_score([1.0, 0.0], [False, True]) == 1.0


def test_brier_intermediate() -> None:
    brier = compute_brier_score([0.8, 0.6, 0.4], [True, True, False])
    assert 0.0 < brier < 1.0


def test_brier_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_brier_score([], [])


def test_brier_mismatched_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_brier_score([0.5], [True, False])


# compute_calibration -------------------------------------------------------


def test_calibration_with_evaluation_records() -> None:
    evals = [
        _make_eval(0.9, True),
        _make_eval(0.8, True),
        _make_eval(0.7, False),
        _make_eval(0.2, False),
        _make_eval(0.6, True),
    ]
    result = compute_calibration(evals, n_bins=5)
    assert isinstance(result, CalibrationResult)
    assert result.n_samples == 5
    assert result.n_bins == 5
    assert 0.0 <= result.ece <= 1.0
    assert 0.0 <= result.mce <= 1.0
    assert 0.0 <= result.brier_score <= 1.0
    assert len(result.bin_accuracies) == 5
    assert len(result.bin_confidences) == 5
    assert len(result.bin_counts) == 5


def test_calibration_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_calibration([])


def test_calibration_single_record() -> None:
    evals = [_make_eval(1.0, True)]
    result = compute_calibration(evals)
    assert result.n_samples == 1
    assert result.n_bins == 10


def test_calibration_all_high_confidence_correct() -> None:
    evals = [_make_eval(0.95, True) for _ in range(20)]
    result = compute_calibration(evals, n_bins=5)
    assert result.ece < 0.1


def test_calibration_all_high_confidence_wrong() -> None:
    evals = [_make_eval(0.95, False) for _ in range(20)]
    result = compute_calibration(evals, n_bins=5)
    assert result.ece > 0.8


# _reliability_diagram_data -------------------------------------------------


def test_reliability_diagram_data_shape() -> None:
    evals = [_make_eval(s / 10, s >= 5) for s in range(11)]
    data = _reliability_diagram_data(evals, n_bins=5)
    assert set(data) == {"bin_accuracies", "bin_confidences", "bin_counts", "ece", "mce", "n_bins"}
    assert len(data["bin_accuracies"]) == 5
    assert len(data["bin_confidences"]) == 5
    assert len(data["bin_counts"]) == 5
    assert 0.0 <= data["ece"] <= 1.0
    assert 0.0 <= data["mce"] <= 1.0


# Edge cases ----------------------------------------------------------------


def test_calibration_large_n_bins() -> None:
    evals = [_make_eval(0.5, i % 2 == 0) for i in range(100)]
    result = compute_calibration(evals, n_bins=50)
    assert result.n_bins == 50
    assert len(result.bin_counts) == 50


def test_calibration_all_zeros() -> None:
    evals = [_make_eval(0.0, False) for _ in range(10)]
    result = compute_calibration(evals, n_bins=5)
    assert result.ece == 0.0
    assert result.brier_score == 0.0


def test_calibration_all_ones() -> None:
    evals = [_make_eval(1.0, True) for _ in range(10)]
    result = compute_calibration(evals, n_bins=5)
    assert result.ece == 0.0
    assert result.brier_score == 0.0


def test_calibration_metadata_is_empty() -> None:
    evals = [_make_eval(0.5, True)]
    result = compute_calibration(evals)
    assert result.metadata == {}
