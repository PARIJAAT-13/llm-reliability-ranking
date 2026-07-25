"""Tests for compute_isr (Information Survival Rate)."""

import pytest

from llm_reliability.metrics.isr import compute_isr, compute_temporal_isr
from tests.metrics_helpers import make_eval


def test_isr_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_isr([])


def test_isr_no_faulted_raises():
    evs = [make_eval("t1", success=True, score=1.0, fault_injected=False)]
    with pytest.raises(ValueError, match="No fault-injected"):
        compute_isr(evs)


def test_isr_perfect():
    """All baseline and faulted scores are 1.0 → ISR = 1.0."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t1", success=True, score=1.0, fault_injected=True),
    ]
    result = compute_isr(evs)
    assert result["isr_output"] == pytest.approx(1.0, abs=1e-9)
    assert result["isr_behavior"] == pytest.approx(1.0, abs=1e-9)
    assert result["isr_composite"] == pytest.approx(1.0, abs=1e-9)


def test_isr_zero():
    """Baseline scores = 1.0, faulted scores = 0.0 → ISR = 0.0 (no overlap)."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t1", success=True, score=0.0, fault_injected=True),
    ]
    result = compute_isr(evs)
    assert result["isr_output"] == pytest.approx(0.0, abs=1e-9)
    assert result["isr_behavior"] == pytest.approx(0.0, abs=1e-9)


def test_isr_partial_behavior():
    """Half the scores preserved under fault → behavior ISR = 0.5."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t2", success=True, score=1.0, fault_injected=False),
        make_eval("t1", success=True, score=0.5, fault_injected=True),
        make_eval("t2", success=True, score=0.5, fault_injected=True),
    ]
    result = compute_isr(evs)
    # sr_base=1.0, sr_fault=0.5 → 1 - |1-0.5| = 0.5
    assert result["isr_behavior"] == pytest.approx(0.5, abs=1e-9)


def test_isr_partial_output():
    """Mixture of preserved and degraded scores → intermediate output ISR."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t2", success=True, score=1.0, fault_injected=False),
        make_eval("t1", success=True, score=0.0, fault_injected=True),
        make_eval("t2", success=True, score=1.0, fault_injected=True),
    ]
    result = compute_isr(evs, n_bins=2)
    # With n_bins=2, bins are [0, 0.5] and [0.5, 1]
    # Baseline: 2 scores in [0.5, 1], 0 in [0, 0.5]
    # Faulted: 1 in [0, 0.5], 1 in [0.5, 1]
    # Intersection = min(0/2, 1/2)/bin + min(2/2, 1/2)/bin
    # Normalised by bin_width=0.5:
    # p_base = [0, 1] (density), p_fault = [0.5/dw, 0.5/dw] where dw=0.5
    # Actually let me just test it's in range and > 0 < 1
    assert 0.0 < result["isr_output"] < 1.0


def test_isr_composite_default_alpha():
    """ISR composite = 0.6 * output + 0.4 * behavior with default alpha."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t1", success=True, score=0.0, fault_injected=True),
    ]
    result = compute_isr(evs)
    expected = 0.6 * result["isr_output"] + 0.4 * result["isr_behavior"]
    assert result["isr_composite"] == pytest.approx(expected, abs=1e-9)


def test_isr_composite_custom_alpha():
    """Custom alpha should affect composite weighting."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t1", success=True, score=0.0, fault_injected=True),
    ]
    result = compute_isr(evs, alpha=0.2)
    expected = 0.2 * result["isr_output"] + 0.8 * result["isr_behavior"]
    assert result["isr_composite"] == pytest.approx(expected, abs=1e-9)


def test_isr_in_range():
    """All ISR components must be in [0, 1]."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t2", success=False, score=0.0, fault_injected=False),
        make_eval("t1", success=True, score=0.3, fault_injected=True),
        make_eval("t2", success=False, score=0.7, fault_injected=True),
    ]
    result = compute_isr(evs)
    assert 0.0 <= result["isr_output"] <= 1.0
    assert 0.0 <= result["isr_behavior"] <= 1.0
    assert 0.0 <= result["isr_composite"] <= 1.0


def test_isr_deterministic():
    """ISR must be deterministic for the same inputs."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t2", success=False, score=0.0, fault_injected=False),
        make_eval("t1", success=True, score=0.3, fault_injected=True),
        make_eval("t2", success=False, score=0.7, fault_injected=True),
    ]
    result1 = compute_isr(evs)
    result2 = compute_isr(evs)
    assert result1["isr_output"] == result2["isr_output"]
    assert result1["isr_behavior"] == result2["isr_behavior"]
    assert result1["isr_composite"] == result2["isr_composite"]


def test_isr_custom_n_bins():
    """Different n_bins may yield different output ISR."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t1", success=True, score=0.5, fault_injected=True),
        make_eval("t2", success=True, score=0.5, fault_injected=True),
    ]
    result_2 = compute_isr(evs, n_bins=2)
    result_10 = compute_isr(evs, n_bins=10)
    # Different bin counts → different histogram resolution
    assert result_2["isr_output"] != result_10["isr_output"]


def test_isr_per_fault_type():
    """Per-fault-type ISR should be returned when fault names are present."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t2", success=True, score=1.0, fault_injected=False),
        make_eval("t1", success=True, score=0.5, fault_injected=True),
        make_eval("t2", success=True, score=0.5, fault_injected=True),
    ]
    result = compute_isr(evs)
    assert "per_fault_type" in result
    assert "unknown" in result["per_fault_type"]


def test_isr_counts():
    """Result should include baseline and fault counts."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t2", success=True, score=1.0, fault_injected=False),
        make_eval("t1", success=True, score=0.5, fault_injected=True),
    ]
    result = compute_isr(evs)
    assert result["n_baseline"] == 2
    assert result["n_fault"] == 1


def test_isr_no_baseline_fault_only():
    """ISR with baseline being empty due to only faulted records."""
    evs = [
        make_eval("t1", success=True, score=0.5, fault_injected=True),
        make_eval("t2", success=True, score=0.5, fault_injected=True),
    ]
    result = compute_isr(evs)
    assert result["isr_output"] == 0.0
    assert 0.0 <= result["isr_behavior"] <= 1.0
    assert 0.0 <= result["isr_composite"] <= 1.0


# ---------------------------------------------------------------------------
# Bootstrap CI tests
# ---------------------------------------------------------------------------


def test_isr_bootstrap_ci_in_result():
    """Bootstrap CI keys present when ci_method='bootstrap'."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t2", success=True, score=1.0, fault_injected=False),
        make_eval("t3", success=True, score=0.8, fault_injected=False),
        make_eval("t1", success=True, score=0.5, fault_injected=True),
        make_eval("t2", success=True, score=0.6, fault_injected=True),
        make_eval("t3", success=True, score=0.7, fault_injected=True),
    ]
    result = compute_isr(evs, ci_method="bootstrap", n_resamples=50, random_seed=42)
    assert "isr_output_ci" in result
    assert "isr_behavior_ci" in result
    assert result["isr_output_ci"] is not None
    assert result["isr_behavior_ci"] is not None


def test_isr_bootstrap_ci_is_interval():
    """Bootstrap CI should return (lower, upper) with lower <= upper."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t2", success=True, score=1.0, fault_injected=False),
        make_eval("t3", success=True, score=0.8, fault_injected=False),
        make_eval("t1", success=True, score=0.5, fault_injected=True),
        make_eval("t2", success=True, score=0.6, fault_injected=True),
        make_eval("t3", success=True, score=0.7, fault_injected=True),
    ]
    result = compute_isr(evs, ci_method="bootstrap", n_resamples=50, random_seed=0)
    lo, hi = result["isr_output_ci"]
    assert lo <= hi
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


def test_isr_bootstrap_deterministic():
    """Bootstrap with same seed should give same CI."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t2", success=True, score=0.9, fault_injected=False),
        make_eval("t3", success=True, score=0.8, fault_injected=False),
        make_eval("t1", success=True, score=0.5, fault_injected=True),
        make_eval("t2", success=True, score=0.4, fault_injected=True),
        make_eval("t3", success=True, score=0.6, fault_injected=True),
    ]
    r1 = compute_isr(evs, ci_method="bootstrap", n_resamples=50, random_seed=123)
    r2 = compute_isr(evs, ci_method="bootstrap", n_resamples=50, random_seed=123)
    assert r1["isr_output_ci"] == r2["isr_output_ci"]
    assert r1["isr_behavior_ci"] == r2["isr_behavior_ci"]


def test_isr_bootstrap_none_by_default():
    """Without ci_method='bootstrap', CI fields should be None."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t2", success=True, score=1.0, fault_injected=False),
        make_eval("t1", success=True, score=0.5, fault_injected=True),
    ]
    result = compute_isr(evs)
    assert result["isr_output_ci"] is None
    assert result["isr_behavior_ci"] is None


def test_isr_bootstrap_too_few_samples():
    """Bootstrap CI should return None when < 2 baseline or fault samples."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t1", success=True, score=0.5, fault_injected=True),
    ]
    result = compute_isr(evs, ci_method="bootstrap", n_resamples=50)
    assert result["isr_output_ci"] is None
    assert result["isr_behavior_ci"] is None


# ---------------------------------------------------------------------------
# Temporal ISR tests
# ---------------------------------------------------------------------------


def test_temporal_isr_basic():
    """Temporal ISR returns correct structure."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t2", success=True, score=1.0, fault_injected=False),
    ]
    for i in range(10):
        evs.append(
            make_eval(
                f"t{i}",
                success=True,
                score=0.8,
                fault_injected=True,
                seed=i + 10,
            )
        )
    result = compute_temporal_isr(evs, n_windows=3)
    assert "window_isr" in result
    assert "trend_slope" in result
    assert "overall_isr" in result
    assert len(result["window_isr"]) == 3
    assert 0.0 <= result["overall_isr"] <= 1.0


def test_temporal_isr_degradation():
    """Simulate degradation: later windows should have lower ISR."""
    evs = [
        make_eval("t_base", success=True, score=1.0, fault_injected=False),
    ]
    # First 5: high scores (ISR near 1)
    for i in range(5):
        evs.append(make_eval(f"t_early_{i}", success=True, score=0.95, fault_injected=True, seed=i))
    # Last 5: low scores (ISR lower)
    for i in range(5):
        evs.append(
            make_eval(f"t_late_{i}", success=True, score=0.3, fault_injected=True, seed=100 + i)
        )
    result = compute_temporal_isr(evs, n_windows=2)
    assert len(result["window_isr"]) == 2
    # First window should be higher ISR than second
    assert result["window_isr"][0] > result["window_isr"][1]
    # Trend slope should be negative (degradation)
    assert result["trend_slope"] < 0


def test_temporal_isr_empty_raises():
    """Empty evaluations should raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        compute_temporal_isr([])


def test_temporal_isr_no_faulted_raises():
    """No fault-injected records should raise ValueError."""
    evs = [make_eval("t1", success=True, score=1.0, fault_injected=False)]
    with pytest.raises(ValueError, match="No fault-injected"):
        compute_temporal_isr(evs)


def test_temporal_isr_few_windows_raises():
    """n_windows < 2 should raise ValueError."""
    evs = [
        make_eval("t1", success=True, score=1.0, fault_injected=False),
        make_eval("t1", success=True, score=0.5, fault_injected=True),
    ]
    with pytest.raises(ValueError, match="n_windows must be >= 2"):
        compute_temporal_isr(evs, n_windows=1)


def test_temporal_isr_labels():
    """Window labels should be (start, end) index tuples."""
    evs = [
        make_eval("t_base", success=True, score=1.0, fault_injected=False),
    ]
    for i in range(6):
        evs.append(make_eval(f"t{i}", success=True, score=0.8, fault_injected=True, seed=i))
    result = compute_temporal_isr(evs, n_windows=3)
    assert len(result["window_labels"]) == 3
    for label in result["window_labels"]:
        assert isinstance(label, tuple)
        assert len(label) == 2
        assert label[0] < label[1] or label[0] == label[1]
