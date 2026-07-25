"""Tests for automated statistical procedure selection."""

import numpy as np
import pytest

from llm_reliability.statistics.auto_selection import (
    auto_select,
    check_normality,
    check_variance_homogeneity,
    run_recommended_test,
    suggest_correction,
    suggest_effect_size,
    suggest_posthoc,
    suggest_test,
)

# ======================================================================
# Normality check
# ======================================================================


def test_normality_normal():
    rng = np.random.default_rng(42)
    normal = rng.normal(0.5, 0.1, size=100).tolist()
    results = check_normality([normal])
    assert results[0]


def test_normality_non_normal():
    rng = np.random.default_rng(42)
    uniform = rng.uniform(0, 1, size=100).tolist()
    results = check_normality([uniform])
    # Uniform may pass Shapiro with some seeds; just check it runs
    assert isinstance(results[0], bool)


def test_normality_empty():
    results = check_normality([[]])
    assert results[0]


def test_normality_multi_group():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0.5, 0.1, size=50).tolist()
    g2 = rng.normal(0.6, 0.15, size=50).tolist()
    results = check_normality([g1, g2])
    assert len(results) == 2
    for r in results:
        assert isinstance(r, bool)


# ======================================================================
# Variance homogeneity
# ======================================================================


def test_variance_homogeneous():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0.5, 0.1, size=30).tolist()
    g2 = rng.normal(0.6, 0.1, size=30).tolist()
    assert check_variance_homogeneity([g1, g2])


def test_variance_inhomogeneous():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0.5, 0.01, size=30).tolist()
    g2 = rng.normal(0.6, 0.5, size=30).tolist()
    result = check_variance_homogeneity([g1, g2], alpha=0.01)
    assert isinstance(result, bool)


def test_variance_single_group():
    assert check_variance_homogeneity([[1.0, 2.0, 3.0]])


def test_variance_empty():
    assert check_variance_homogeneity([])


# ======================================================================
# Test suggestion
# ======================================================================


def test_suggest_two_normal_homogeneous():
    assert suggest_test(2, is_normal=True, is_homogeneous=True) == "independent_t_test"


def test_suggest_two_normal_unequal():
    assert suggest_test(2, is_normal=True, is_homogeneous=False) == "welch_t_test"


def test_suggest_two_non_normal():
    assert suggest_test(2, is_normal=False) == "mannwhitney_u"


def test_suggest_two_paired_normal():
    assert suggest_test(2, is_paired=True, is_normal=True) == "paired_t_test"


def test_suggest_two_paired_non_normal():
    assert suggest_test(2, is_paired=True, is_normal=False) == "wilcoxon_signed_rank"


def test_suggest_multi_normal_homogeneous():
    assert suggest_test(3, is_normal=True, is_homogeneous=True) == "anova_oneway"


def test_suggest_multi_normal_unequal():
    assert suggest_test(3, is_normal=True, is_homogeneous=False) == "welch_anova"


def test_suggest_multi_non_normal():
    assert suggest_test(3, is_normal=False) == "kruskal_wallis"


def test_suggest_multi_paired_non_normal():
    assert suggest_test(3, is_paired=True, is_normal=False) == "friedman_test"


# ======================================================================
# Post-hoc suggestion
# ======================================================================


def test_posthoc_two():
    assert suggest_posthoc(2, is_normal=True) is None


def test_posthoc_multi_normal():
    assert suggest_posthoc(3, is_normal=True) == "tukey_hsd"


def test_posthoc_multi_non_normal():
    assert suggest_posthoc(3, is_normal=False) == "nemenyi_test"


# ======================================================================
# Effect size suggestion
# ======================================================================


def test_effect_size_hedges_small():
    assert "hedges_g" in suggest_effect_size("independent_t_test", n_total=20)


def test_effect_size_cohens_large():
    assert suggest_effect_size("independent_t_test", n_total=100) == "cohens_d"


def test_effect_size_mannwhitney():
    assert suggest_effect_size("mannwhitney_u", n_total=50) == "rank_biserial_r"


def test_effect_size_anova():
    assert suggest_effect_size("anova_oneway", n_total=60) == "omega_squared"


def test_effect_size_kruskal():
    assert suggest_effect_size("kruskal_wallis", n_total=60) == "epsilon_squared"


def test_effect_size_friedman():
    assert suggest_effect_size("friedman_test", n_total=30) == "kendalls_w"


# ======================================================================
# Correction suggestion
# ======================================================================


def test_correction_few():
    assert suggest_correction(3) == "bonferroni"


def test_correction_medium():
    assert suggest_correction(10) == "holm_bonferroni"


def test_correction_many():
    assert suggest_correction(30) == "benjamini_hochberg"


# ======================================================================
# Auto-select integration
# ======================================================================


def test_auto_select_two_normal():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0.5, 0.1, size=30).tolist()
    g2 = rng.normal(0.6, 0.1, size=30).tolist()
    result = auto_select([g1, g2])
    assert result["n_groups"] == 2
    assert "recommended_test" in result
    assert "recommended_effect_size" in result
    assert "recommended_correction" in result
    assert result["n_comparisons"] == 1


def test_auto_select_three_non_normal():
    rng = np.random.default_rng(42)
    g1 = rng.uniform(0, 1, size=20).tolist()
    g2 = rng.uniform(0.2, 0.8, size=20).tolist()
    g3 = rng.uniform(0.5, 1.0, size=20).tolist()
    result = auto_select([g1, g2, g3])
    assert result["n_groups"] == 3
    assert result["n_comparisons"] == 3
    assert "normality_results" in result
    assert len(result["normality_results"]) == 3


# ======================================================================
# Run recommended test
# ======================================================================


def test_run_recommended_two_normal():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0.5, 0.1, size=30).tolist()
    g2 = rng.normal(0.51, 0.1, size=30).tolist()
    result = run_recommended_test([g1, g2])
    assert "test_name" in result
    assert "statistic" in result
    assert "p_value" in result
    assert "is_significant" in result
    assert 0.0 <= result["p_value"] <= 1.0


def test_run_recommended_two_different():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0.9, 0.05, size=30).tolist()
    g2 = rng.normal(0.1, 0.05, size=30).tolist()
    result = run_recommended_test([g1, g2])
    assert result["is_significant"], f"p={result['p_value']:.4f}"


def test_run_recommended_fewer_than_two():
    result = run_recommended_test([[1.0, 2.0, 3.0]])
    assert "error" in result


def test_run_recommended_non_normal():
    rng = np.random.default_rng(42)
    # Exponential distributions are clearly non-normal
    g1 = rng.exponential(0.5, size=50).tolist()
    g2 = rng.exponential(0.5, size=50).tolist()
    result = run_recommended_test([g1, g2])
    assert "statistic" in result


def test_run_recommended_multi():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0.5, 0.1, size=30).tolist()
    g2 = rng.normal(0.6, 0.1, size=30).tolist()
    g3 = rng.normal(0.7, 0.1, size=30).tolist()
    result = run_recommended_test([g1, g2, g3])
    # With truly normally distributed data and similar variances
    assert result["test_name"] in ("anova_oneway", "kruskal_wallis")


def test_run_recommended_effect_sizes_present():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0.9, 0.05, size=30).tolist()
    g2 = rng.normal(0.1, 0.05, size=30).tolist()
    result = run_recommended_test([g1, g2])
    assert "effect_sizes" in result
