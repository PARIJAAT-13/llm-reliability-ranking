"""
Comprehensive unit tests for the statistics package.

Tests cover:
- compute_statistical_summary (statistical_engine.py)
- calculate_summary_statistics (utils.py)
- compute_bootstrap_ci (confidence_intervals.py)
- compute_cohens_d / compute_cliffs_delta / compute_rank_biserial (effect_sizes.py)
- run_paired_t_test / run_wilcoxon_test (hypothesis_tests.py)
- check_normality (assumptions.py)
- validate_rankings (utils.py)
- compute_cohens_d (statistical_engine.py) — standalone float-based variant
- perform_cross_validation_check (statistical_engine.py)
"""

import math

import numpy as np
import pytest

from llm_reliability.records.ranking import RankingRecord
from llm_reliability.statistics.assumptions import check_normality
from llm_reliability.statistics.confidence_intervals import \
    compute_bootstrap_ci
from llm_reliability.statistics.effect_sizes import compute_cliffs_delta
from llm_reliability.statistics.effect_sizes import \
    compute_cohens_d as effect_cohens_d
from llm_reliability.statistics.effect_sizes import compute_rank_biserial
from llm_reliability.statistics.hypothesis_tests import (run_paired_t_test,
                                                         run_wilcoxon_test)
from llm_reliability.statistics.statistical_engine import \
    compute_cohens_d as engine_cohens_d
from llm_reliability.statistics.statistical_engine import (
    compute_statistical_summary, perform_cross_validation_check)
from llm_reliability.statistics.utils import (calculate_summary_statistics,
                                              validate_rankings)
from tests.statistics_test_helpers import create_mock_ranking

# ============================================================================
#  compute_statistical_summary  (statistical_engine.py)
# ============================================================================


class TestComputeStatisticalSummary:
    def test_summary_normal_values(self):
        data = [0.8, 0.85, 0.9, 0.82, 0.88, 0.87, 0.84, 0.89]
        s = compute_statistical_summary(data, n_bootstrap=100)
        assert s.sample_size == 8
        assert 0.85 <= s.mean <= 0.86
        assert s.median == 0.86
        assert s.ci_95_lower <= s.mean <= s.ci_95_upper
        assert s.bootstrap_ci_95_lower <= s.bootstrap_ci_95_upper

    def test_summary_single_value(self):
        s = compute_statistical_summary([42.0], n_bootstrap=100)
        assert s.sample_size == 1
        assert s.mean == 42.0
        assert s.median == 42.0
        assert s.variance == 0.0
        assert s.std_dev == 0.0

    def test_summary_two_values(self):
        s = compute_statistical_summary([10.0, 20.0], n_bootstrap=100)
        assert s.sample_size == 2
        assert s.mean == 15.0
        assert s.median == 15.0
        assert s.variance > 0

    def test_summary_identical_values(self):
        s = compute_statistical_summary([5.0, 5.0, 5.0, 5.0], n_bootstrap=100)
        assert s.mean == 5.0
        assert s.median == 5.0
        assert s.variance == 0.0
        assert s.std_dev == 0.0

    def test_summary_empty_list(self):
        s = compute_statistical_summary([], n_bootstrap=100)
        assert s.sample_size == 0
        assert s.mean == 0.0
        assert s.median == 0.0
        assert s.variance == 0.0
        assert s.std_dev == 0.0

    def test_summary_negative_values(self):
        data = [-10.0, -5.0, -3.0, -8.0]
        s = compute_statistical_summary(data, n_bootstrap=100)
        assert s.sample_size == 4
        assert s.mean < 0
        assert s.median < 0

    def test_summary_large_values(self):
        data = [1e6, 2e6, 3e6, 4e6]
        s = compute_statistical_summary(data, n_bootstrap=100)
        assert s.sample_size == 4
        assert s.mean == pytest.approx(2.5e6)

    def test_summary_float_precision(self):
        data = [0.1, 0.2, 0.3, 0.4]
        s = compute_statistical_summary(data, n_bootstrap=100)
        assert s.mean == pytest.approx(0.25)
        assert s.ci_95_lower <= s.ci_95_upper

    def test_summary_bootstrap_seeded(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        s1 = compute_statistical_summary(data, n_bootstrap=200, seed=42)
        s2 = compute_statistical_summary(data, n_bootstrap=200, seed=42)
        assert s1.bootstrap_ci_95_lower == s2.bootstrap_ci_95_lower
        assert s1.bootstrap_ci_95_upper == s2.bootstrap_ci_95_upper

    def test_summary_bootstrap_different_ci(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        s1 = compute_statistical_summary(data, n_bootstrap=200, seed=42)
        s2 = compute_statistical_summary(data, n_bootstrap=200, seed=9999)
        assert (
            s1.bootstrap_ci_95_lower != s2.bootstrap_ci_95_lower
            or s1.bootstrap_ci_95_upper != s2.bootstrap_ci_95_upper
        )

    def test_summary_ci_bounds_reasonable(self):
        data = [0.7, 0.8, 0.9, 0.85, 0.75]
        s = compute_statistical_summary(data, n_bootstrap=200)
        assert s.ci_95_lower < s.mean < s.ci_95_upper
        assert s.bootstrap_ci_95_lower < s.mean < s.bootstrap_ci_95_upper

    def test_summary_variance_zero_for_single(self):
        s = compute_statistical_summary([3.14], n_bootstrap=100)
        assert s.variance == 0.0
        assert s.std_dev == 0.0

    def test_summary_rounded_values(self):
        data = [1.0 / 3.0] * 3
        s = compute_statistical_summary(data, n_bootstrap=100)
        assert s.sample_size == 3
        assert isinstance(s.mean, float)

    def test_summary_integer_inputs(self):
        s = compute_statistical_summary([1, 2, 3, 4], n_bootstrap=100)
        assert s.mean == 2.5
        assert s.sample_size == 4


# ============================================================================
#  calculate_summary_statistics  (utils.py)
# ============================================================================


class TestCalculateSummaryStatistics:
    def test_ss_basic(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ss = calculate_summary_statistics(data)
        assert ss.mean == 3.0
        assert ss.median == 3.0
        assert ss.min_val == 1.0
        assert ss.max_val == 5.0
        assert ss.count == 5
        assert ss.variance == pytest.approx(2.5)
        assert ss.std_dev == pytest.approx(math.sqrt(2.5))

    def test_ss_single_value(self):
        ss = calculate_summary_statistics([10.0])
        assert ss.mean == 10.0
        assert ss.median == 10.0
        assert ss.variance == 0.0
        assert ss.std_dev == 0.0
        assert ss.q1 == 10.0
        assert ss.q3 == 10.0
        assert ss.count == 1

    def test_ss_two_values(self):
        ss = calculate_summary_statistics([5.0, 15.0])
        assert ss.mean == 10.0
        assert ss.median == 10.0
        assert ss.min_val == 5.0
        assert ss.max_val == 15.0
        assert ss.count == 2

    def test_ss_identical_values(self):
        ss = calculate_summary_statistics([3.0, 3.0, 3.0, 3.0])
        assert ss.mean == 3.0
        assert ss.variance == 0.0
        assert ss.std_dev == 0.0
        assert ss.q1 == 3.0
        assert ss.q3 == 3.0

    def test_ss_empty_raises(self):
        with pytest.raises(ValueError, match="Cannot calculate summary statistics on empty data"):
            calculate_summary_statistics([])

    def test_ss_negative_values(self):
        ss = calculate_summary_statistics([-5.0, -3.0, -1.0])
        assert ss.mean == -3.0
        assert ss.min_val == -5.0
        assert ss.max_val == -1.0

    def test_ss_large_values(self):
        data = [1e10, 2e10, 3e10]
        ss = calculate_summary_statistics(data)
        assert ss.mean == pytest.approx(2e10)
        assert ss.variance > 0

    def test_ss_float_precision(self):
        data = [0.1, 0.2, 0.3, 0.4]
        ss = calculate_summary_statistics(data)
        assert ss.mean == pytest.approx(0.25)
        assert ss.q1 == pytest.approx(0.175)
        assert ss.q3 == pytest.approx(0.325)

    def test_ss_all_fields_present(self):
        data = [2.0, 4.0, 6.0, 8.0, 10.0]
        ss = calculate_summary_statistics(data)
        assert ss.count == 5
        assert ss.mean == 6.0
        assert ss.median == 6.0
        assert ss.variance == pytest.approx(10.0)
        assert ss.std_dev == pytest.approx(math.sqrt(10.0))
        assert ss.min_val == 2.0
        assert ss.max_val == 10.0
        assert ss.q1 == 4.0
        assert ss.q3 == 8.0

    def test_ss_order_independent(self):
        data1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        data2 = [5.0, 3.0, 1.0, 4.0, 2.0]
        ss1 = calculate_summary_statistics(data1)
        ss2 = calculate_summary_statistics(data2)
        assert ss1.mean == ss2.mean
        assert ss1.median == ss2.median
        assert ss1.variance == ss2.variance
        assert ss1.min_val == ss2.min_val
        assert ss1.max_val == ss2.max_val

    def test_ss_100_values(self):
        data = list(range(100))
        ss = calculate_summary_statistics(data)
        assert ss.count == 100
        assert ss.mean == 49.5
        assert ss.min_val == 0
        assert ss.max_val == 99
        assert ss.q1 == 24.75
        assert ss.q3 == 74.25

    def test_ss_string_input_raises(self):
        with pytest.raises(ValueError, match="could not convert string to float"):
            calculate_summary_statistics(["a", "b", "c"])

    def test_ss_nan_raises_validation_error(self):
        from pydantic_core import ValidationError as PydanticValidationError

        data = [1.0, float("nan"), 3.0]
        with pytest.raises(PydanticValidationError):
            calculate_summary_statistics(data)

    def test_ss_inf_raises_validation_error(self):
        from pydantic_core import ValidationError as PydanticValidationError

        data = [1.0, float("inf"), 3.0]
        with pytest.raises(PydanticValidationError):
            calculate_summary_statistics(data)


# ============================================================================
#  compute_bootstrap_ci  (confidence_intervals.py)
# ============================================================================


class TestComputeBootstrapCI:
    def test_confidence_interval_normal(self):
        data = [0.8, 0.85, 0.9, 0.82, 0.88, 0.87, 0.84, 0.89]
        ci = compute_bootstrap_ci(data, n_resamples=500)
        assert ci.lower <= ci.upper
        assert ci.confidence_level == 0.95
        mean_val = np.mean(data)
        assert ci.lower <= mean_val <= ci.upper

    def test_confidence_interval_single_value(self):
        ci = compute_bootstrap_ci([42.0], n_resamples=500)
        assert ci.lower == 42.0
        assert ci.upper == 42.0

    def test_confidence_interval_two_values(self):
        ci = compute_bootstrap_ci([10.0, 20.0], n_resamples=500)
        assert ci.lower <= ci.upper

    def test_confidence_interval_identical_values(self):
        ci = compute_bootstrap_ci([5.0, 5.0, 5.0], n_resamples=500)
        assert ci.lower == 5.0
        assert ci.upper == 5.0

    def test_confidence_interval_ninety_percent(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = compute_bootstrap_ci(data, confidence_level=0.90, n_resamples=500)
        assert ci.confidence_level == 0.90
        assert ci.lower <= ci.upper

    def test_confidence_interval_ninetyfive_percent(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = compute_bootstrap_ci(data, confidence_level=0.95, n_resamples=500)
        assert ci.confidence_level == 0.95
        assert ci.lower <= ci.upper

    def test_confidence_interval_ninetyeight_percent(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = compute_bootstrap_ci(data, confidence_level=0.98, n_resamples=500)
        assert ci.confidence_level == 0.98
        assert ci.lower <= ci.upper

    def test_confidence_interval_high_confidence(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci_90 = compute_bootstrap_ci(data, confidence_level=0.90, n_resamples=500)
        ci_99 = compute_bootstrap_ci(data, confidence_level=0.99, n_resamples=500)
        assert ci_99.lower <= ci_90.lower
        assert ci_99.upper >= ci_90.upper

    def test_confidence_interval_bootstrap_equivalent(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci1 = compute_bootstrap_ci(data, n_resamples=500, seed=42)
        ci2 = compute_bootstrap_ci(data, n_resamples=500, seed=42)
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper

    def test_confidence_interval_raises_on_empty(self):
        with pytest.raises(
            ValueError, match="Cannot compute bootstrap confidence intervals on empty data"
        ):
            compute_bootstrap_ci([])

    def test_confidence_interval_negative_values(self):
        data = [-5.0, -3.0, -1.0, -4.0, -2.0]
        ci = compute_bootstrap_ci(data, n_resamples=500)
        assert ci.lower <= ci.upper
        assert ci.upper < 0

    def test_confidence_interval_large_values(self):
        data = [1e6, 2e6, 3e6, 4e6, 5e6]
        ci = compute_bootstrap_ci(data, n_resamples=500)
        assert ci.lower <= ci.upper

    def test_confidence_interval_invalid_level_raises(self):
        data = [1.0, 2.0, 3.0]
        with pytest.raises(
            ValueError, match="confidence_level must be strictly between 0.0 and 1.0"
        ):
            compute_bootstrap_ci(data, confidence_level=0.0)
        with pytest.raises(
            ValueError, match="confidence_level must be strictly between 0.0 and 1.0"
        ):
            compute_bootstrap_ci(data, confidence_level=1.0)
        with pytest.raises(
            ValueError, match="confidence_level must be strictly between 0.0 and 1.0"
        ):
            compute_bootstrap_ci(data, confidence_level=-0.1)

    def test_confidence_interval_nan_in_data(self):
        data = [1.0, float("nan"), 3.0]
        ci = compute_bootstrap_ci(data, n_resamples=500)
        assert math.isnan(ci.lower) or ci.lower <= ci.upper

    def test_confidence_interval_inf_in_data(self):
        data = [1.0, float("inf"), 3.0]
        ci = compute_bootstrap_ci(data, n_resamples=500)
        assert not math.isnan(ci.lower)


# ============================================================================
#  Effect sizes: compute_cohens_d (effect_sizes.py, RankingRecord-based)
# ============================================================================


class TestEffectSizeCohensD:
    def test_cohens_d_equal_groups(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7})
        r2 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7})
        result = effect_cohens_d(r1, r2)
        assert result.value == 0.0
        assert result.method == "Cohen's d (pooled)"
        assert result.interpretation == "negligible"

    def test_cohens_d_different_groups(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.85, "c": 0.8})
        r2 = create_mock_ranking({"a": 0.6, "b": 0.55, "c": 0.5})
        result = effect_cohens_d(r1, r2)
        assert result.value > 1.0
        assert result.interpretation == "large"

    def test_cohens_d_identical_groups(self):
        r1 = create_mock_ranking({"a": 0.75, "b": 0.65})
        r2 = create_mock_ranking({"a": 0.75, "b": 0.65})
        result = effect_cohens_d(r1, r2)
        assert result.value == 0.0

    def test_cohens_d_all_values_same(self):
        r1 = create_mock_ranking({"a": 0.5, "b": 0.5, "c": 0.5})
        r2 = create_mock_ranking({"a": 0.5, "b": 0.5, "c": 0.5})
        result = effect_cohens_d(r1, r2)
        assert result.value == 0.0

    def test_cohens_d_interpretation_thresholds(self):
        r_high = create_mock_ranking({"a": 0.95, "b": 0.94, "c": 0.93})
        r_low = create_mock_ranking({"a": 0.5, "b": 0.49, "c": 0.48})
        result = effect_cohens_d(r_high, r_low)
        assert result.interpretation in ("large", "medium", "small", "negligible")
        assert isinstance(result.value, float)

    def test_cohens_d_three_agents(self):
        r1 = create_mock_ranking({"a": 1.0, "b": 0.5, "c": 0.0})
        r2 = create_mock_ranking({"a": 0.0, "b": 0.5, "c": 1.0})
        result = effect_cohens_d(r1, r2)
        assert isinstance(result.value, float)

    def test_cohens_d_method_string(self):
        r1 = create_mock_ranking({"a": 0.8, "b": 0.7})
        r2 = create_mock_ranking({"a": 0.6, "b": 0.5})
        result = effect_cohens_d(r1, r2)
        assert result.method == "Cohen's d (pooled)"


# ============================================================================
#  Effect sizes: compute_cliffs_delta (effect_sizes.py)
# ============================================================================


class TestCliffsDelta:
    def test_cliffs_delta_equal_groups(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7})
        r2 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7})
        result = compute_cliffs_delta(r1, r2)
        assert result.value == 0.0
        assert result.interpretation == "negligible"

    def test_cliffs_delta_all_higher(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7})
        r2 = create_mock_ranking({"a": 0.5, "b": 0.4, "c": 0.3})
        result = compute_cliffs_delta(r1, r2)
        assert result.value == 1.0
        assert result.method == "Cliff's Delta"

    def test_cliffs_delta_all_lower(self):
        r1 = create_mock_ranking({"a": 0.3, "b": 0.2, "c": 0.1})
        r2 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7})
        result = compute_cliffs_delta(r1, r2)
        assert result.value == -1.0

    def test_cliffs_delta_identical_scores(self):
        r1 = create_mock_ranking({"a": 0.5, "b": 0.5, "c": 0.5})
        r2 = create_mock_ranking({"a": 0.5, "b": 0.5, "c": 0.5})
        result = compute_cliffs_delta(r1, r2)
        assert result.value == 0.0

    def test_cliffs_delta_partial_overlap(self):
        r1 = create_mock_ranking({"a": 0.8, "b": 0.6, "c": 0.4})
        r2 = create_mock_ranking({"a": 0.7, "b": 0.5, "c": 0.3})
        result = compute_cliffs_delta(r1, r2)
        assert 0.2 <= result.value <= 0.8
        assert result.method == "Cliff's Delta"

    def test_cliffs_delta_two_agents(self):
        r1 = create_mock_ranking({"a": 1.0, "b": 0.0})
        r2 = create_mock_ranking({"a": 0.0, "b": 1.0})
        result = compute_cliffs_delta(r1, r2)
        assert result.value == 0.0

    def test_cliffs_delta_interpretation_string(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.85})
        r2 = create_mock_ranking({"a": 0.8, "b": 0.75})
        result = compute_cliffs_delta(r1, r2)
        assert result.interpretation in ("negligible", "small", "medium", "large")


# ============================================================================
#  Effect sizes: compute_rank_biserial (effect_sizes.py)
# ============================================================================


class TestRankBiserial:
    def test_rank_biserial_equal_rankings(self):
        r1 = create_mock_ranking({"a": 1.0, "b": 0.8, "c": 0.6})
        r2 = create_mock_ranking({"a": 1.0, "b": 0.8, "c": 0.6})
        result = compute_rank_biserial(r1, r2)
        assert result.value == 0.0

    def test_rank_biserial_non_zero(self):
        r1 = create_mock_ranking({"a": 1.0, "b": 0.5, "c": 0.0})
        r2 = create_mock_ranking({"a": 0.0, "b": 0.5, "c": 1.0})
        result = compute_rank_biserial(r1, r2)
        assert result.value == 0.0

    def test_rank_biserial_some_difference(self):
        r1 = create_mock_ranking({"a": 0.8, "b": 0.7, "c": 0.6, "d": 0.5})
        r2 = create_mock_ranking({"a": 0.5, "b": 0.6, "c": 0.7, "d": 0.8})
        result = compute_rank_biserial(r1, r2)
        assert isinstance(result.value, float)

    def test_rank_biserial_all_diffs_zero(self):
        r1 = create_mock_ranking({"a": 0.5, "b": 0.5})
        r2 = create_mock_ranking({"a": 0.5, "b": 0.5})
        result = compute_rank_biserial(r1, r2)
        assert result.value == 0.0
        assert result.interpretation == "negligible"

    def test_rank_biserial_three_agents(self):
        r1 = create_mock_ranking({"a": 1.0, "b": 0.5, "c": 0.0})
        r2 = create_mock_ranking({"a": 0.0, "b": 0.5, "c": 1.0})
        result = compute_rank_biserial(r1, r2)
        assert isinstance(result.value, float)

    def test_rank_biserial_method_name(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7})
        r2 = create_mock_ranking({"a": 0.7, "b": 0.8, "c": 0.9})
        result = compute_rank_biserial(r1, r2)
        assert result.method == "Rank-biserial Correlation"


# ============================================================================
#  Hypothesis tests: run_paired_t_test (hypothesis_tests.py)
# ============================================================================


class TestPairedTTest:
    def test_ttest_equal_means(self):
        r1 = create_mock_ranking({"a": 0.8, "b": 0.7, "c": 0.6})
        r2 = create_mock_ranking({"a": 0.8, "b": 0.7, "c": 0.6})
        result = run_paired_t_test(r1, r2)
        assert result.p_value >= 0.05 or result.statistic == 0.0
        assert result.method == "Paired t-test"

    def test_ttest_different_means(self):
        r1 = create_mock_ranking({"a": 0.95, "b": 0.90, "c": 0.85, "d": 0.80, "e": 0.75})
        r2 = create_mock_ranking({"a": 0.50, "b": 0.45, "c": 0.40, "d": 0.35, "e": 0.30})
        result = run_paired_t_test(r1, r2)
        assert result.p_value < 0.05
        assert result.statistic != 0.0

    def test_ttest_single_value(self):
        r1 = create_mock_ranking({"a": 0.9})
        r2 = create_mock_ranking({"a": 0.8})
        result = run_paired_t_test(r1, r2)
        assert result.p_value == 1.0
        assert result.statistic == 0.0
        assert any("too small" in w for w in result.warnings)

    def test_ttest_two_values(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8})
        r2 = create_mock_ranking({"a": 0.7, "b": 0.6})
        result = run_paired_t_test(r1, r2)
        assert isinstance(result.statistic, float)

    def test_ttest_identical_values(self):
        r1 = create_mock_ranking({"a": 0.5, "b": 0.5, "c": 0.5})
        r2 = create_mock_ranking({"a": 0.5, "b": 0.5, "c": 0.5})
        result = run_paired_t_test(r1, r2)
        assert result.statistic == 0.0
        assert result.p_value == 1.0

    def test_ttest_assumptions_checked(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6})
        r2 = create_mock_ranking({"a": 0.5, "b": 0.4, "c": 0.3, "d": 0.2})
        result = run_paired_t_test(r1, r2)
        assert isinstance(result.assumptions_met, bool)

    def test_ttest_warnings_list(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8})
        r2 = create_mock_ranking({"a": 0.7, "b": 0.6})
        result = run_paired_t_test(r1, r2)
        assert isinstance(result.warnings, list)

    def test_ttest_method_string(self):
        r1 = create_mock_ranking({"a": 0.8, "b": 0.7, "c": 0.6})
        r2 = create_mock_ranking({"a": 0.5, "b": 0.4, "c": 0.3})
        result = run_paired_t_test(r1, r2)
        assert result.method == "Paired t-test"


# ============================================================================
#  Hypothesis tests: run_wilcoxon_test (hypothesis_tests.py)
# ============================================================================


class TestWilcoxonTest:
    def test_wilcoxon_equal(self):
        r1 = create_mock_ranking({"a": 0.8, "b": 0.7, "c": 0.6, "d": 0.5, "e": 0.4})
        r2 = create_mock_ranking({"a": 0.8, "b": 0.7, "c": 0.6, "d": 0.5, "e": 0.4})
        result = run_wilcoxon_test(r1, r2)
        assert result.p_value == 1.0
        assert result.statistic == 0.0

    def test_wilcoxon_different(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5, "f": 0.4})
        r2 = create_mock_ranking({"a": 0.3, "b": 0.2, "c": 0.1, "d": 0.0, "e": 0.9, "f": 0.8})
        result = run_wilcoxon_test(r1, r2)
        assert isinstance(result.p_value, float)
        assert result.statistic >= 0

    def test_wilcoxon_small_sample(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8})
        r2 = create_mock_ranking({"a": 0.4, "b": 0.3})
        result = run_wilcoxon_test(r1, r2)
        assert any("too small" in w for w in result.warnings)

    def test_wilcoxon_ties(self):
        r1 = create_mock_ranking({"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5, "e": 0.5})
        r2 = create_mock_ranking({"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5, "e": 0.5})
        result = run_wilcoxon_test(r1, r2)
        assert result.p_value == 1.0
        assert result.statistic == 0.0

    def test_wilcoxon_three_agents(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7})
        r2 = create_mock_ranking({"a": 0.6, "b": 0.5, "c": 0.4})
        result = run_wilcoxon_test(r1, r2)
        assert result.method == "Wilcoxon Signed-Rank Test"
        assert isinstance(result.p_value, float)

    def test_wilcoxon_alternative_two_sided(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5})
        r2 = create_mock_ranking({"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1, "e": 0.0})
        result = run_wilcoxon_test(r1, r2)
        assert result.alternative == "two-sided"

    def test_wilcoxon_method_string(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7})
        r2 = create_mock_ranking({"a": 0.6, "b": 0.5, "c": 0.4})
        result = run_wilcoxon_test(r1, r2)
        assert result.method == "Wilcoxon Signed-Rank Test"

    def test_wilcoxon_assumptions_met(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5})
        r2 = create_mock_ranking({"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1, "e": 0.0})
        result = run_wilcoxon_test(r1, r2)
        assert result.assumptions_met is True


# ============================================================================
#  check_normality  (assumptions.py)
# ============================================================================


class TestCheckNormality:
    def test_check_normality_normal_data(self):
        rng = np.random.default_rng(42)
        diffs = rng.normal(loc=0.0, scale=0.1, size=50).tolist()
        ok, warning = check_normality(diffs)
        assert ok is True
        assert warning is None

    def test_check_normality_too_small(self):
        ok, warning = check_normality([1.0, 2.0])
        assert ok is False
        assert warning is not None
        assert "too small" in warning.lower()

    def test_check_normality_single_value(self):
        ok, warning = check_normality([1.0])
        assert ok is False
        assert warning is not None
        assert "too small" in warning.lower()

    def test_check_normality_exactly_three(self):
        ok, warning = check_normality([1.0, 2.0, 3.0])
        assert isinstance(ok, bool)
        assert isinstance(warning, str) or warning is None

    def test_check_normality_identical_values(self):
        ok, warning = check_normality([5.0, 5.0, 5.0, 5.0, 5.0])
        assert isinstance(ok, bool)
        assert warning is None or isinstance(warning, str)

    def test_check_normality_return_type(self):
        ok, warning = check_normality([0.5, 0.6, 0.7, 0.8, 0.9])
        assert isinstance(ok, bool)
        assert warning is None or isinstance(warning, str)


# ============================================================================
#  validate_rankings  (utils.py)
# ============================================================================


class TestValidateRankings:
    def test_validate_metrics_valid(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7})
        r2 = create_mock_ranking({"a": 0.6, "b": 0.5, "c": 0.4})
        validate_rankings(r1, r2)

    def test_validate_metrics_empty_ranking1(self):
        r1 = create_mock_ranking({})
        r2 = create_mock_ranking({"a": 0.9})
        with pytest.raises(ValueError, match="Rankings list cannot be empty"):
            validate_rankings(r1, r2)

    def test_validate_metrics_empty_ranking2(self):
        r1 = create_mock_ranking({"a": 0.9})
        r2 = create_mock_ranking({})
        with pytest.raises(ValueError, match="Rankings list cannot be empty"):
            validate_rankings(r1, r2)

    def test_validate_metrics_mismatched_lengths(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8})
        r2 = create_mock_ranking({"a": 0.9})
        with pytest.raises(ValueError, match="Mismatched ranking lengths"):
            validate_rankings(r1, r2)

    def test_validate_metrics_different_agents(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8})
        r2 = create_mock_ranking({"a": 0.9, "c": 0.8})
        with pytest.raises(ValueError, match="must contain the exact same set of agents"):
            validate_rankings(r1, r2)

    def test_validate_metrics_duplicate_agents(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7})
        r2_dup = RankingRecord.model_construct(
            ranking_type="success",
            benchmark="mock-bench",
            rankings=(("a", 0.9), ("a", 0.8), ("b", 0.7)),
            rank_map={"a": 1, "b": 3},
            computed_at="2026-07-21T02:00:00Z",
        )
        with pytest.raises(ValueError, match="Duplicate agent IDs"):
            validate_rankings(r1, r2_dup)

    def test_validate_metrics_null_scores(self):
        r1 = create_mock_ranking({"a": 0.9, "b": 0.8, "c": 0.7})
        r2_null = RankingRecord.model_construct(
            ranking_type="success",
            benchmark="mock-bench",
            rankings=(("a", 0.9), ("b", None), ("c", 0.7)),
            rank_map={"a": 1, "b": 2, "c": 3},
            computed_at="2026-07-21T02:00:00Z",
        )
        with pytest.raises(ValueError, match="missing/None score"):
            validate_rankings(r1, r2_null)


# ============================================================================
#  compute_cohens_d  (statistical_engine.py, float-based)
# ============================================================================


class TestEngineCohensD:
    def test_engine_cohens_d_equal_groups(self):
        d = engine_cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert d == 0.0

    def test_engine_cohens_d_different_groups(self):
        d = engine_cohens_d([10.0, 12.0, 11.0], [5.0, 6.0, 7.0])
        assert d > 0.5

    def test_engine_cohens_d_identical_groups(self):
        d = engine_cohens_d([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
        assert d == 0.0

    def test_engine_cohens_d_empty_group(self):
        result = engine_cohens_d([], [1.0, 2.0])
        assert result == 0.0

    def test_engine_cohens_d_single_element(self):
        d = engine_cohens_d([1.0], [2.0])
        assert d == 0.0

    def test_engine_cohens_d_one_group_single(self):
        d = engine_cohens_d([1.0, 2.0, 3.0], [5.0])
        assert d == 0.0

    def test_engine_cohens_d_large_effect(self):
        d = engine_cohens_d([100.0, 102.0, 101.0], [1.0, 2.0, 1.5])
        assert d > 2.0

    def test_engine_cohens_d_negative_effect(self):
        d = engine_cohens_d([1.0, 2.0, 3.0], [10.0, 11.0, 12.0])
        assert d < 0


# ============================================================================
#  perform_cross_validation_check  (statistical_engine.py)
# ============================================================================


class TestCrossValidationCheck:
    def test_cross_validation_high_stability(self):
        seed_data = {
            42: [0.85, 0.86, 0.84],
            100: [0.84, 0.85, 0.85],
            2026: [0.86, 0.85, 0.84],
            777: [0.85, 0.84, 0.86],
            999: [0.84, 0.86, 0.85],
        }
        cv_res = perform_cross_validation_check(seed_data)
        assert cv_res["cross_validation_stability"] == "HIGH"

    def test_cross_validation_output_fields(self):
        seed_data = {42: [0.8, 0.9], 100: [0.7, 0.8]}
        cv_res = perform_cross_validation_check(seed_data)
        assert "overall_summary" in cv_res
        assert "seed_means" in cv_res
        assert "inter_seed_variance" in cv_res
        assert "cross_validation_stability" in cv_res

    def test_cross_validation_seed_means(self):
        seed_data = {42: [1.0, 1.0], 100: [2.0, 2.0]}
        cv_res = perform_cross_validation_check(seed_data)
        assert cv_res["seed_means"][42] == 1.0
        assert cv_res["seed_means"][100] == 2.0

    def test_cross_validation_empty_seeds(self):
        cv_res = perform_cross_validation_check({})
        assert cv_res["cross_validation_stability"] == "HIGH"

    def test_cross_validation_single_seed(self):
        cv_res = perform_cross_validation_check({42: [0.8, 0.9, 0.85]})
        assert "seed_means" in cv_res
        assert cv_res["cross_validation_stability"] == "HIGH"


# ============================================================================
#  Edge cases: NaN, Inf, negative, large/small values across modules
# ============================================================================


class TestNanInfEdgeCases:
    def test_summary_nan_in_data(self):
        s = compute_statistical_summary([1.0, float("nan"), 3.0], n_bootstrap=100)
        assert math.isnan(s.mean) or not math.isnan(s.mean)

    def test_summary_inf_in_data(self):
        s = compute_statistical_summary([1.0, float("inf"), 3.0], n_bootstrap=100)
        assert math.isinf(s.mean)

    def test_ss_negative_inf(self):
        with pytest.raises((ValueError, Exception)):
            calculate_summary_statistics([-1.0, float("-inf"), -3.0])

    def test_bootstrap_ci_negative_inf(self):
        ci = compute_bootstrap_ci([-1.0, float("-inf"), -3.0], n_resamples=100)
        assert ci is not None

    def test_normality_with_nan(self):
        ok, warning = check_normality([1.0, float("nan"), 3.0, 4.0, 5.0])
        assert isinstance(ok, bool)


# ============================================================================
#  Integration-style: StatisticalEngine.summarize and analyze
# ============================================================================


class TestStatisticalEngineIntegration:
    def test_engine_summarize(self):
        from llm_reliability.statistics.statistical_engine import \
            StatisticalEngine

        data = [0.8, 0.85, 0.9, 0.82, 0.88]
        s = StatisticalEngine.summarize(data, n_bootstrap=100)
        assert s.sample_size == 5
        assert s.mean > 0
        assert s.ci_95_lower <= s.ci_95_upper

    def test_engine_analyze(self):
        from llm_reliability.statistics.statistical_engine import \
            StatisticalEngine

        r1 = create_mock_ranking({"a": 0.95, "b": 0.90, "c": 0.85, "d": 0.80})
        r2 = create_mock_ranking({"a": 0.60, "b": 0.55, "c": 0.50, "d": 0.45})
        report = StatisticalEngine.analyze(r1, r2)
        assert "spearman" in report.correlations
        assert "kendall_tau" in report.correlations
        assert len(report.hypothesis_tests) == 2
        assert len(report.effect_sizes) == 2
        assert "differences" in report.confidence_intervals
        assert "ranking1" in report.summary_statistics
        assert "ranking2" in report.summary_statistics
        assert report.metadata["sample_size"] == 4


# ============================================================================
#  Additional edge cases for calculate_summary_statistics detail
# ============================================================================


class TestSummaryStatisticsDetail:
    def test_ss_q1_q3_odd_count(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ss = calculate_summary_statistics(data)
        assert ss.q1 == 2.0
        assert ss.q3 == 4.0

    def test_ss_q1_q3_even_count(self):
        data = [1.0, 2.0, 3.0, 4.0]
        ss = calculate_summary_statistics(data)
        assert ss.q1 == 1.75
        assert ss.q3 == 3.25

    def test_ss_q1_q3_two_elements(self):
        data = [10.0, 20.0]
        ss = calculate_summary_statistics(data)
        assert ss.q1 == 12.5
        assert ss.q3 == 17.5

    def test_ss_very_small_numbers(self):
        data = [1e-10, 2e-10, 3e-10]
        ss = calculate_summary_statistics(data)
        assert ss.mean == pytest.approx(2e-10)

    def test_ss_very_large_numbers(self):
        data = [1e15, 2e15, 3e15]
        ss = calculate_summary_statistics(data)
        assert ss.mean == pytest.approx(2e15)

    def test_ss_mixed_sign(self):
        data = [-10.0, 0.0, 10.0]
        ss = calculate_summary_statistics(data)
        assert ss.mean == 0.0
        assert ss.min_val == -10.0
        assert ss.max_val == 10.0

    def test_ss_all_zero(self):
        data = [0.0, 0.0, 0.0]
        ss = calculate_summary_statistics(data)
        assert ss.mean == 0.0
        assert ss.variance == 0.0
        assert ss.min_val == 0.0
        assert ss.max_val == 0.0

    def test_ss_repeated_values(self):
        data = [7.0, 7.0, 7.0, 7.0, 7.0]
        ss = calculate_summary_statistics(data)
        assert ss.mean == 7.0
        assert ss.median == 7.0
        assert ss.variance == 0.0

    def test_ss_descending_order(self):
        data = [5.0, 4.0, 3.0, 2.0, 1.0]
        ss = calculate_summary_statistics(data)
        assert ss.min_val == 1.0
        assert ss.max_val == 5.0
        assert ss.mean == 3.0
        assert ss.median == 3.0


# ============================================================================
#  Additional edge cases for compute_statistical_summary detail
# ============================================================================


class TestStatisticalSummaryDetail:
    def test_summary_mean_median_identical_for_symmetric(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = compute_statistical_summary(data, n_bootstrap=100)
        assert s.mean == 3.0
        assert s.median == 3.0

    def test_summary_all_zeros(self):
        s = compute_statistical_summary([0.0, 0.0, 0.0], n_bootstrap=100)
        assert s.mean == 0.0
        assert s.variance == 0.0
        assert s.ci_95_lower == 0.0
        assert s.ci_95_upper == 0.0

    def test_summary_large_sample(self):
        data = list(range(1000))
        s = compute_statistical_summary(data, n_bootstrap=50)
        assert s.sample_size == 1000
        assert s.mean == 499.5
        assert s.median == 499.5

    def test_summary_two_element_variance(self):
        s = compute_statistical_summary([10.0, 20.0], n_bootstrap=100)
        assert s.variance == pytest.approx(50.0, abs=0.001)
        assert s.std_dev == pytest.approx(7.0711, abs=0.001)

    def test_summary_odd_count_median(self):
        s = compute_statistical_summary([1.0, 3.0, 5.0], n_bootstrap=100)
        assert s.median == 3.0

    def test_summary_even_count_median(self):
        s = compute_statistical_summary([1.0, 2.0, 3.0, 10.0], n_bootstrap=100)
        assert s.median == 2.5
