"""
Unit tests for Statistical Engine.
"""

from llm_reliability.statistics.statistical_engine import (
    compute_cohens_d,
    compute_statistical_summary,
    perform_cross_validation_check,
)


def test_statistical_summary():
    data = [0.8, 0.85, 0.9, 0.82, 0.88, 0.87, 0.84, 0.89]
    summary = compute_statistical_summary(data, n_bootstrap=100)
    assert summary.sample_size == 8
    assert summary.mean > 0.8
    assert summary.bootstrap_ci_95_lower <= summary.mean <= summary.bootstrap_ci_95_upper


def test_cohens_d():
    g1 = [0.9, 0.92, 0.88, 0.91]
    g2 = [0.6, 0.65, 0.58, 0.62]
    d = compute_cohens_d(g1, g2)
    assert d > 1.0  # Large effect size


def test_cross_validation_check():
    seed_data = {
        42: [0.85, 0.86, 0.84],
        100: [0.84, 0.85, 0.85],
        2026: [0.86, 0.85, 0.84],
        777: [0.85, 0.84, 0.86],
        999: [0.84, 0.86, 0.85],
    }
    cv_res = perform_cross_validation_check(seed_data)
    assert cv_res["cross_validation_stability"] == "HIGH"
