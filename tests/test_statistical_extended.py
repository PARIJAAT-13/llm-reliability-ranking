"""Extended tests for statistical engine — edge cases, corrections, and auto-selection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from llm_reliability.records.ranking import RankingRecord
from llm_reliability.statistics.assumptions import check_normality
from llm_reliability.statistics.auto_selection import (
    auto_select,
    check_variance_homogeneity,
    run_recommended_test,
    suggest_correction,
    suggest_test,
)
from llm_reliability.statistics.correlation import compute_kendall_tau, compute_spearman
from llm_reliability.statistics.effect_sizes import (
    compute_cliffs_delta,
    compute_cohens_d,
    compute_rank_biserial,
)
from llm_reliability.statistics.extensions import (
    benjamini_hochberg_correction,
    bonferroni_correction,
    compute_bayes_factor_ttest,
    compute_eta_squared,
    compute_glasss_delta,
    compute_hedges_g,
    compute_omega_squared,
    compute_posthoc_power,
    compute_required_sample_size,
    holm_bonferroni_correction,
    run_friedman_test,
    run_kruskal_wallis,
    run_mannwhitney_u,
    run_nemenyi_posthoc,
    run_oneway_anova,
)
from llm_reliability.statistics.statistical_engine import (
    StatisticalEngine,
)
from llm_reliability.statistics.statistical_engine import (
    compute_cohens_d as engine_cohens_d,
)
from llm_reliability.statistics.statistical_engine import (
    compute_statistical_summary,
    perform_cross_validation_check,
)

TIMESTAMP = "2026-01-01T00:00:00+00:00"


def _make_ranking(
    agent_scores: tuple[tuple[str, float], ...], rtype: str = "success"
) -> RankingRecord:
    rank_map = {agent: i + 1 for i, (agent, _) in enumerate(agent_scores)}
    return RankingRecord(
        ranking_type=rtype,
        benchmark="test",
        rankings=agent_scores,
        rank_map=rank_map,
        computed_at=TIMESTAMP,
    )


class TestStatisticalEngineEdgeCases:
    def test_summary_empty_data(self):
        summary = compute_statistical_summary([])
        assert summary.sample_size == 0
        assert summary.mean == 0.0
        assert summary.median == 0.0
        assert summary.variance == 0.0

    def test_summary_single_value(self):
        summary = compute_statistical_summary([5.0], n_bootstrap=100)
        assert summary.sample_size == 1
        assert summary.mean == 5.0
        assert summary.median == 5.0

    def test_summary_two_values(self):
        summary = compute_statistical_summary([3.0, 7.0], n_bootstrap=100)
        assert summary.sample_size == 2
        assert summary.mean == 5.0
        assert summary.median == 5.0

    def test_summary_all_same_value(self):
        summary = compute_statistical_summary([1.0, 1.0, 1.0, 1.0], n_bootstrap=100)
        assert summary.mean == 1.0
        assert summary.variance == 0.0
        assert summary.std_dev == 0.0

    def test_cross_validation_empty(self):
        result = perform_cross_validation_check({})
        assert result["overall_summary"]["sample_size"] == 0

    def test_cross_validation_single_seed(self):
        result = perform_cross_validation_check({42: [0.8, 0.9, 0.85]})
        assert result["overall_summary"]["sample_size"] == 3


class TestBootstrapCI:
    def test_bootstrap_ci_small_sample(self):
        data = [0.5, 0.6]
        summary = compute_statistical_summary(data, n_bootstrap=500)
        assert summary.bootstrap_ci_95_lower <= summary.bootstrap_ci_95_upper
        assert summary.sample_size == 2

    def test_bootstrap_ci_larger_sample(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0.75, 0.1, size=50).tolist()
        summary = compute_statistical_summary(data, n_bootstrap=500)
        assert summary.bootstrap_ci_95_lower <= summary.mean <= summary.bootstrap_ci_95_upper


class TestCorrelation:
    def test_spearman_perfect_positive(self):
        r1 = _make_ranking((("a", 1.0), ("b", 2.0), ("c", 3.0)))
        r2 = _make_ranking((("a", 1.0), ("b", 2.0), ("c", 3.0)))
        result = compute_spearman(r1, r2)
        assert result.coefficient == pytest.approx(1.0)

    def test_spearman_perfect_negative(self):
        r1 = _make_ranking((("a", 1.0), ("b", 2.0), ("c", 3.0)))
        r2 = _make_ranking((("a", 3.0), ("b", 2.0), ("c", 1.0)))
        result = compute_spearman(r1, r2)
        assert result.coefficient == pytest.approx(-1.0)

    def test_spearman_tied_ranks(self):
        r1 = _make_ranking((("a", 1.0), ("b", 1.0), ("c", 2.0)))
        r2 = _make_ranking((("a", 2.0), ("b", 2.0), ("c", 1.0)))
        result = compute_spearman(r1, r2)
        assert -1.0 <= result.coefficient <= 1.0

    def test_spearman_two_agents(self):
        r1 = _make_ranking((("a", 1.0), ("b", 2.0)))
        r2 = _make_ranking((("a", 2.0), ("b", 1.0)))
        result = compute_spearman(r1, r2)
        assert result.coefficient == pytest.approx(-1.0)

    def test_kendall_tau_perfect_positive(self):
        r1 = _make_ranking((("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)))
        r2 = _make_ranking((("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)))
        result = compute_kendall_tau(r1, r2)
        assert result.coefficient == pytest.approx(1.0, abs=0.01)

    def test_kendall_tau_perfect_negative(self):
        r1 = _make_ranking((("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)))
        r2 = _make_ranking((("a", 4.0), ("b", 3.0), ("c", 2.0), ("d", 1.0)))
        result = compute_kendall_tau(r1, r2)
        assert result.coefficient == pytest.approx(-1.0, abs=0.01)

    def test_kendall_tau_tied_ranks(self):
        r1 = _make_ranking((("a", 1.0), ("b", 1.0), ("c", 2.0), ("d", 2.0)))
        r2 = _make_ranking((("a", 2.0), ("b", 2.0), ("c", 1.0), ("d", 1.0)))
        result = compute_kendall_tau(r1, r2)
        assert -1.0 <= result.coefficient <= 1.0


class TestEffectSizes:
    def test_cohens_d_identical_groups(self):
        r1 = _make_ranking((("a", 0.8), ("b", 0.8)))
        r2 = _make_ranking((("a", 0.8), ("b", 0.8)))
        d = compute_cohens_d(r1, r2)
        assert d.value == 0.0

    def test_cohens_d_different_groups(self):
        r1 = _make_ranking((("a", 0.9), ("b", 0.92)))
        r2 = _make_ranking((("a", 0.6), ("b", 0.65)))
        d = compute_cohens_d(r1, r2)
        assert d.value != 0.0

    def test_cohens_d_small_samples(self):
        r1 = _make_ranking((("a", 0.5), ("b", 0.6)))
        r2 = _make_ranking((("a", 0.8), ("b", 0.7)))
        d = compute_cohens_d(r1, r2)
        assert d.value != 0.0

    def test_cliffs_delta_identical_groups(self):
        r1 = _make_ranking((("a", 0.5), ("b", 0.6)))
        r2 = _make_ranking((("a", 0.5), ("b", 0.6)))
        delta = compute_cliffs_delta(r1, r2)
        assert delta.value == 0.0

    def test_cliffs_delta_no_overlap(self):
        r1 = _make_ranking((("a", 0.9), ("b", 1.0)))
        r2 = _make_ranking((("a", 0.1), ("b", 0.2)))
        delta = compute_cliffs_delta(r1, r2)
        assert delta.value == pytest.approx(1.0)

    def test_rank_biserial_identical(self):
        r1 = _make_ranking((("a", 0.5), ("b", 0.6)))
        r2 = _make_ranking((("a", 0.5), ("b", 0.6)))
        r = compute_rank_biserial(r1, r2)
        assert r.value == 0.0

    def test_hedges_g_identical(self):
        g = compute_hedges_g([1.0, 2.0], [1.0, 2.0])
        assert g.value == 0.0

    def test_hedges_g_small(self):
        g = compute_hedges_g([1.0], [2.0])
        assert g.value == 0.0

    def test_hedges_g_different(self):
        g = compute_hedges_g([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert g.value != 0.0

    def test_glasss_delta_identical(self):
        delta = compute_glasss_delta([1.0, 2.0], [1.0, 2.0])
        assert delta.value == 0.0

    def test_glasss_delta_empty(self):
        delta = compute_glasss_delta([], [1.0])
        assert delta.value == 0.0

    def test_eta_squared_identical_groups(self):
        eta = compute_eta_squared([1.0, 2.0], [1.0, 2.0])
        assert eta.value == 0.0

    def test_eta_squared_different(self):
        eta = compute_eta_squared([1.0, 2.0], [5.0, 6.0])
        assert eta.value > 0.5

    def test_eta_squared_single_group(self):
        eta = compute_eta_squared([1.0, 2.0, 3.0])
        assert eta.value == 0.0

    def test_omega_squared_identical(self):
        omega = compute_omega_squared([1.0, 2.0], [1.0, 2.0])
        assert omega.value == 0.0

    def test_omega_squared_single_group(self):
        omega = compute_omega_squared([1.0, 2.0, 3.0])
        assert omega.value == 0.0


class TestStatisticalEngineAnalyze:
    def test_analyze_identical_rankings(self):
        r1 = _make_ranking((("a", 0.9), ("b", 0.8), ("c", 0.7)))
        r2 = _make_ranking((("a", 0.9), ("b", 0.8), ("c", 0.7)))
        report = StatisticalEngine.analyze(r1, r2)
        assert report.correlations["spearman"].coefficient == pytest.approx(1.0)
        assert report.correlations["kendall_tau"].coefficient == pytest.approx(1.0)

    def test_analyze_reversed_rankings(self):
        r1 = _make_ranking((("a", 0.9), ("b", 0.8), ("c", 0.7)))
        r2 = _make_ranking((("a", 0.7), ("b", 0.8), ("c", 0.9)))
        report = StatisticalEngine.analyze(r1, r2)
        assert report.correlations["spearman"].coefficient == pytest.approx(-1.0)

    def test_summarize(self):
        summary = StatisticalEngine.summarize([0.8, 0.85, 0.9, 0.82])
        assert summary.sample_size == 4
        assert summary.mean == pytest.approx(0.8425, abs=0.01)


class TestAutoSelection:
    def test_auto_select_two_normal(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(0.5, 0.1, size=30).tolist()
        g2 = rng.normal(0.5, 0.1, size=30).tolist()
        result = auto_select([g1, g2])
        assert result["n_groups"] == 2
        assert "recommended_test" in result
        assert "recommended_correction" in result

    def test_auto_select_three_groups(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(0.5, 0.1, size=20).tolist()
        g2 = rng.normal(0.6, 0.1, size=20).tolist()
        g3 = rng.normal(0.7, 0.1, size=20).tolist()
        result = auto_select([g1, g2, g3])
        assert result["n_groups"] == 3
        assert result["n_comparisons"] == 3

    def test_auto_select_single_group(self):
        result = auto_select([[1.0, 2.0, 3.0]])
        assert result["n_groups"] == 1
        assert result["recommended_test"] == "no_test"

    def test_auto_select_non_normal(self):
        rng = np.random.default_rng(42)
        g1 = rng.exponential(0.5, size=50).tolist()
        g2 = rng.exponential(0.5, size=50).tolist()
        result = auto_select([g1, g2])
        assert result["n_groups"] == 2

    def test_run_recommended_large_effect(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(0.9, 0.05, size=30).tolist()
        g2 = rng.normal(0.1, 0.05, size=30).tolist()
        result = run_recommended_test([g1, g2])
        assert result["is_significant"]

    def test_run_recommended_small_effect(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(0.5, 0.1, size=30).tolist()
        g2 = rng.normal(0.51, 0.1, size=30).tolist()
        result = run_recommended_test([g1, g2])
        assert "statistic" in result


class TestMultipleComparisonCorrection:
    def test_bonferroni_all_significant(self):
        p_values = [0.001, 0.002, 0.003]
        result = bonferroni_correction(p_values, alpha=0.05)
        assert all(result)

    def test_bonferroni_none_significant(self):
        p_values = [0.5, 0.6, 0.7]
        result = bonferroni_correction(p_values, alpha=0.05)
        assert not any(result)

    def test_bonferroni_empty(self):
        result = bonferroni_correction([], alpha=0.05)
        assert result == []

    def test_holm_bonferroni_all_significant(self):
        p_values = [0.001, 0.002, 0.003]
        result = holm_bonferroni_correction(p_values, alpha=0.05)
        assert all(result)

    def test_holm_bonferroni_none_significant(self):
        p_values = [0.5, 0.6, 0.7]
        result = holm_bonferroni_correction(p_values, alpha=0.05)
        assert not any(result)

    def test_holm_bonferroni_empty(self):
        result = holm_bonferroni_correction([], alpha=0.05)
        assert result == []

    def test_holm_bonferroni_partial(self):
        p_values = [0.001, 0.04, 0.5]
        result = holm_bonferroni_correction(p_values, alpha=0.05)
        assert result[0] is True
        assert result[2] is False

    def test_benjamini_hochberg_all_significant(self):
        p_values = [0.001, 0.002, 0.003]
        result = benjamini_hochberg_correction(p_values, alpha=0.05)
        assert all(result)

    def test_benjamini_hochberg_none_significant(self):
        p_values = [0.5, 0.6, 0.7]
        result = benjamini_hochberg_correction(p_values, alpha=0.05)
        assert not any(result)

    def test_benjamini_hochberg_empty(self):
        result = benjamini_hochberg_correction([], alpha=0.05)
        assert result == []

    def test_benjamini_hochberg_partial(self):
        p_values = [0.001, 0.3, 0.02]
        result = benjamini_hochberg_correction(p_values, alpha=0.05)
        assert result[0] is True

    def test_suggest_correction_few(self):
        assert suggest_correction(3) == "bonferroni"

    def test_suggest_correction_medium(self):
        assert suggest_correction(10) == "holm_bonferroni"

    def test_suggest_correction_many(self):
        assert suggest_correction(30) == "benjamini_hochberg"

    def test_suggest_correction_zero(self):
        assert suggest_correction(0) == "bonferroni"


class TestAssumptionsEdgeCases:
    def test_check_normality_less_than_3(self):
        result, msg = check_normality([1.0, 2.0])
        assert result is False
        assert "too small" in msg

    def test_check_normality_normal(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0.5, 0.1, size=50).tolist()
        result, msg = check_normality(data)
        assert result is True
        assert msg is None

    def test_check_normality_empty(self):
        result, msg = check_normality([])
        assert result is False

    def test_variance_homogeneity_empty(self):
        assert check_variance_homogeneity([]) is True

    def test_variance_homogeneity_single(self):
        assert check_variance_homogeneity([[1.0, 2.0, 3.0]]) is True


class TestExtendedHypothesisTests:
    def test_mannwhitney_u_different(self):
        result = run_mannwhitney_u([0.9, 0.8, 0.85, 0.95, 0.88], [0.5, 0.4, 0.45, 0.3, 0.35])
        assert result.p_value < 0.05

    def test_mannwhitney_u_identical(self):
        result = run_mannwhitney_u([0.5, 0.6], [0.5, 0.6])
        assert result.p_value > 0.05

    def test_mannwhitney_u_empty(self):
        result = run_mannwhitney_u([], [1.0, 2.0])
        assert result.statistic == 0.0
        assert result.p_value == 1.0

    def test_mannwhitney_u_very_small(self):
        result = run_mannwhitney_u([0.5], [0.8])
        assert "Very small samples" in " ".join(result.warnings)

    def test_kruskal_wallis_different(self):
        result = run_kruskal_wallis([0.9, 0.8, 0.85], [0.5, 0.4, 0.45], [0.3, 0.2, 0.25])
        assert result.p_value < 0.05

    def test_kruskal_wallis_identical(self):
        result = run_kruskal_wallis([0.5, 0.6], [0.5, 0.6], [0.5, 0.6])
        assert result.p_value > 0.05

    def test_kruskal_wallis_fewer_than_two(self):
        result = run_kruskal_wallis([1.0, 2.0])
        assert result.p_value == 1.0

    def test_friedman_test_different(self):
        result = run_friedman_test([1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0])
        assert result.p_value < 0.05

    def test_friedman_test_identical(self):
        result = run_friedman_test([0.5, 0.6], [0.5, 0.6], [0.5, 0.6])
        assert result.p_value > 0.05

    def test_friedman_test_fewer_than_three(self):
        result = run_friedman_test([1.0, 2.0], [3.0, 4.0])
        assert result.p_value == 1.0

    def test_oneway_anova_different(self):
        result = run_oneway_anova([0.9, 0.8, 0.85], [0.5, 0.4, 0.45])
        assert result.p_value < 0.05

    def test_oneway_anova_identical(self):
        result = run_oneway_anova([0.5, 0.6], [0.5, 0.6])
        assert result.p_value > 0.05

    def test_oneway_anova_fewer_than_two(self):
        result = run_oneway_anova([1.0, 2.0, 3.0])
        assert result.p_value == 1.0

    def test_nemenyi_posthoc_two_groups(self):
        results = run_nemenyi_posthoc([0.9, 0.8, 0.7], [0.5, 0.4, 0.3])
        assert len(results) == 1

    def test_nemenyi_posthoc_fewer_than_two(self):
        results = run_nemenyi_posthoc([1.0, 2.0])
        assert results == []


class TestPowerAnalysis:
    def test_posthoc_power_large_effect(self):
        power = compute_posthoc_power(d=1.0, n=30, alpha=0.05)
        assert power > 0.8

    def test_posthoc_power_no_effect(self):
        power = compute_posthoc_power(d=0.0, n=30, alpha=0.05)
        assert power == 0.0

    def test_posthoc_power_small_n(self):
        power = compute_posthoc_power(d=0.5, n=2, alpha=0.05)
        assert power < 0.1

    def test_required_sample_size(self):
        n = compute_required_sample_size(d=0.5, power=0.8, alpha=0.05)
        assert n > 0

    def test_required_sample_size_zero_d(self):
        n = compute_required_sample_size(d=0.0, power=0.8)
        assert n == 10000


class TestBayesianAnalysis:
    def test_bayes_factor_identical(self):
        result = compute_bayes_factor_ttest([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
        assert result["bf10"] >= 0

    def test_bayes_factor_different(self):
        result = compute_bayes_factor_ttest([0.9, 0.8, 0.85], [0.5, 0.4, 0.45])
        assert "bf10" in result
        assert "interpretation" in result

    def test_bayes_factor_insufficient_data(self):
        result = compute_bayes_factor_ttest([0.5], [0.8])
        assert result["interpretation"] == "insufficient data"

    def test_bayes_factor_mismatched_lengths(self):
        result = compute_bayes_factor_ttest([0.5, 0.6], [0.8])
        assert result["interpretation"] == "insufficient data"


class TestSuggestTest:
    def test_two_normal_homogeneous(self):
        assert suggest_test(2, is_normal=True, is_homogeneous=True) == "independent_t_test"

    def test_two_normal_unequal(self):
        assert suggest_test(2, is_normal=True, is_homogeneous=False) == "welch_t_test"

    def test_two_non_normal(self):
        assert suggest_test(2, is_normal=False) == "mannwhitney_u"

    def test_two_paired_normal(self):
        assert suggest_test(2, is_paired=True, is_normal=True) == "paired_t_test"

    def test_two_paired_non_normal(self):
        assert suggest_test(2, is_paired=True, is_normal=False) == "wilcoxon_signed_rank"

    def test_multi_normal_homogeneous(self):
        assert suggest_test(3, is_normal=True, is_homogeneous=True) == "anova_oneway"

    def test_multi_normal_unequal(self):
        assert suggest_test(3, is_normal=True, is_homogeneous=False) == "welch_anova"

    def test_multi_non_normal(self):
        assert suggest_test(3, is_normal=False) == "kruskal_wallis"

    def test_multi_paired_normal(self):
        assert suggest_test(3, is_paired=True, is_normal=True) == "repeated_measures_anova"

    def test_multi_paired_non_normal(self):
        assert suggest_test(3, is_paired=True, is_normal=False) == "friedman_test"
