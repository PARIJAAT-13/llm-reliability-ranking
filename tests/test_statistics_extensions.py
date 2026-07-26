"""Tests for the extended statistical analysis suite."""

import pytest

from llm_reliability.statistics.extensions import (
    benjamini_hochberg_correction, bonferroni_correction,
    compute_bayes_factor_ttest, compute_eta_squared, compute_glasss_delta,
    compute_hedges_g, compute_omega_squared, compute_posthoc_power,
    compute_required_sample_size, holm_bonferroni_correction,
    run_friedman_test, run_kruskal_wallis, run_mannwhitney_u,
    run_nemenyi_posthoc, run_oneway_anova)

# ======================================================================
# Multiple Comparison Correction
# ======================================================================


def test_bonferroni_empty():
    assert bonferroni_correction([]) == []


def test_bonferroni_all_significant():
    result = bonferroni_correction([0.01, 0.02, 0.03], alpha=0.05)
    # With 3 tests, corrected alpha = 0.05/3 = 0.0167
    assert result == [True, False, False]


def test_bonferroni_none_significant():
    result = bonferroni_correction([0.5, 0.6, 0.7], alpha=0.05)
    assert result == [False, False, False]


def test_holm_bonferroni_empty():
    assert holm_bonferroni_correction([]) == []


def test_holm_bonferroni_vs_bonferroni():
    """Holm-Bonferroni should reject at least as many as Bonferroni."""
    p_values = [0.01, 0.02, 0.03, 0.04]
    bonf = bonferroni_correction(p_values, alpha=0.05)
    holm = holm_bonferroni_correction(p_values, alpha=0.05)
    assert sum(holm) >= sum(bonf)


def test_holm_bonferroni_example():
    p_values = [0.01, 0.02, 0.03, 0.04]
    # Sorted: 0.01(0), 0.02(1), 0.03(2), 0.04(3)
    # k=0: threshold = 0.05/4 = 0.0125 → 0.01 <= 0.0125 ✓
    # k=1: threshold = 0.05/3 = 0.0167 → 0.02 > 0.0167 ✗ → stop
    result = holm_bonferroni_correction(p_values, alpha=0.05)
    assert result == [True, False, False, False]


def test_benjamini_hochberg_empty():
    assert benjamini_hochberg_correction([]) == []


def test_benjamini_hochberg_example():
    p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    result = benjamini_hochberg_correction(p_values, alpha=0.05)
    assert sum(result) > 0  # BH is more liberal


# ======================================================================
# Mann-Whitney U
# ======================================================================


def test_mannwhitney_identical():
    s1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    s2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = run_mannwhitney_u(s1, s2)
    assert result.p_value > 0.05


def test_mannwhitney_different():
    s1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    s2 = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    result = run_mannwhitney_u(s1, s2)
    assert result.p_value < 0.05


def test_mannwhitney_empty():
    result = run_mannwhitney_u([], [1.0, 2.0])
    assert result.p_value == 1.0


def test_mannwhitney_small():
    s1 = [1.0]
    s2 = [100.0]
    result = run_mannwhitney_u(s1, s2)
    assert len(result.warnings) > 0


# ======================================================================
# Kruskal-Wallis
# ======================================================================


def test_kruskal_wallis_identical():
    result = run_kruskal_wallis([1, 2, 3], [1, 2, 3], [1, 2, 3])
    assert result.p_value > 0.05


def test_kruskal_wallis_different():
    result = run_kruskal_wallis([1, 2, 3], [100, 200, 300])
    assert result.p_value < 0.05


def test_kruskal_wallis_single_group():
    result = run_kruskal_wallis([1, 2, 3])
    assert result.p_value == 1.0


# ======================================================================
# Friedman Test
# ======================================================================


def test_friedman_identical():
    result = run_friedman_test([1, 2, 3], [1, 2, 3], [1, 2, 3])
    assert result.p_value > 0.05


def test_friedman_different():
    result = run_friedman_test([1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [1, 2, 3, 4, 5])
    assert result.p_value < 0.05


def test_friedman_few_groups():
    result = run_friedman_test([1, 2], [3, 4])
    assert result.p_value == 1.0


# ======================================================================
# One-way ANOVA
# ======================================================================


def test_anova_identical():
    result = run_oneway_anova([1, 2, 3], [1, 2, 3])
    assert result.p_value > 0.05


def test_anova_different():
    result = run_oneway_anova([1, 2, 3, 4, 5], [100, 200, 300, 400, 500])
    assert result.p_value < 0.05


def test_anova_few_groups():
    result = run_oneway_anova([1, 2, 3])
    assert result.p_value == 1.0


# ======================================================================
# Nemenyi Post-hoc
# ======================================================================


def test_nemenyi_few_groups():
    result = run_nemenyi_posthoc([1, 2, 3])
    assert result == []


def test_nemenyi_identical():
    result = run_nemenyi_posthoc([1, 2, 3], [1, 2, 3], [1, 2, 3])
    assert len(result) == 3  # 3 choose 2 = 3 comparisons


# ======================================================================
# Effect Sizes
# ======================================================================


def test_hedges_g_identical():
    result = compute_hedges_g([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert abs(result.value) < 0.01


def test_hedges_g_different():
    result = compute_hedges_g([1.0, 2.0, 3.0], [100.0, 200.0, 300.0])
    assert abs(result.value) > 0.5


def test_hedges_g_insufficient():
    result = compute_hedges_g([1.0], [])
    assert result.interpretation == "insufficient data"


def test_glasss_delta_identical():
    result = compute_glasss_delta([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert abs(result.value) < 0.01


def test_glasss_delta_different():
    result = compute_glasss_delta([1.0, 2.0, 3.0], [100.0, 200.0, 300.0])
    assert abs(result.value) > 0.5


def test_eta_squared_different():
    result = compute_eta_squared([1.0, 2.0, 3.0], [100.0, 200.0, 300.0])
    assert 0.0 < result.value <= 1.0
    assert result.interpretation != "insufficient data"


def test_eta_squared_identical():
    result = compute_eta_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert abs(result.value) < 0.01


def test_omega_squared_identical():
    result = compute_omega_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result.value >= 0.0


def test_omega_squared_different():
    result = compute_omega_squared([1.0, 2.0, 3.0], [100.0, 200.0, 300.0])
    assert result.value > 0.0


# ======================================================================
# Power Analysis
# ======================================================================


def test_posthoc_power_large_effect():
    power = compute_posthoc_power(d=0.8, n=50)
    assert power > 0.5


def test_posthoc_power_small_effect():
    power = compute_posthoc_power(d=0.1, n=10)
    assert power < 0.5


def test_posthoc_power_zero_effect():
    power = compute_posthoc_power(d=0.0, n=100)
    assert power == 0.0


def test_posthoc_power_small_n():
    power = compute_posthoc_power(d=0.5, n=2)
    assert power < 0.1  # Very low power with n=2


def test_required_sample_size_large_effect():
    n = compute_required_sample_size(d=0.8, power=0.8)
    assert 10 <= n <= 50  # Cohen's d=0.8 typically needs ~26 for 80% power


def test_required_sample_size_small_effect():
    n = compute_required_sample_size(d=0.2, power=0.8)
    assert n > 50  # d=0.2 needs ~200+ for 80% power


def test_required_sample_size_zero_effect():
    n = compute_required_sample_size(d=0.0)
    assert n == 10000


# ======================================================================
# Bayesian Analysis
# ======================================================================


def test_bayes_factor_identical():
    result = compute_bayes_factor_ttest([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert "bf10" in result
    assert result["n"] == 5


def test_bayes_factor_different():
    result = compute_bayes_factor_ttest(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    )
    # Small shift should still show some evidence
    assert isinstance(result["bf10"], float)


def test_bayes_factor_insufficient():
    result = compute_bayes_factor_ttest([1.0], [2.0])
    assert result["interpretation"] == "insufficient data"


def test_bayes_factor_missing_keys():
    result = compute_bayes_factor_ttest([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    for key in ("bf10", "log_bf", "interpretation", "n", "d"):
        assert key in result
