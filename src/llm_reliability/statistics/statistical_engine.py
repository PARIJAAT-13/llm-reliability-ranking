"""
Statistical Expansion & External Validation Engine for LLM Reliability Ranking.

Provides complete parametric and non-parametric statistical metrics, bootstrap confidence
intervals, effect size calculations, hypothesis tests (t-test / Wilcoxon), and multi-seed
cross-validation routines.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

from llm_reliability.utils.serialization import SerializableModel


class StatisticalSummary(SerializableModel):
    """Container for rigorous statistical metrics."""

    mean: float
    median: float
    variance: float
    std_dev: float
    ci_95_lower: float
    ci_95_upper: float
    bootstrap_ci_95_lower: float
    bootstrap_ci_95_upper: float
    sample_size: int


class StatisticalEngine:
    """Engine wrapper for statistical evaluations."""

    @staticmethod
    def summarize(data: Sequence[float], n_bootstrap: int = 1000) -> StatisticalSummary:
        return compute_statistical_summary(data, n_bootstrap=n_bootstrap)

    @staticmethod
    def analyze(ranking1: Any, ranking2: Any) -> Any:
        """Backward-compatible full statistical analysis on two rankings."""
        import numpy as np

        from llm_reliability.statistics.confidence_intervals import \
            compute_bootstrap_ci
        from llm_reliability.statistics.correlation import (
            _align_ranking_scores, compute_kendall_tau, compute_spearman)
        from llm_reliability.statistics.effect_sizes import \
            compute_cliffs_delta
        from llm_reliability.statistics.effect_sizes import \
            compute_cohens_d as calc_cohen
        from llm_reliability.statistics.hypothesis_tests import (
            run_paired_t_test, run_wilcoxon_test)
        from llm_reliability.statistics.result_models import (
            ConfidenceIntervalResult, StatisticalReport, SummaryStatistics)

        spearman = compute_spearman(ranking1, ranking2)
        kendall = compute_kendall_tau(ranking1, ranking2)
        t_test = run_paired_t_test(ranking1, ranking2)
        wilcoxon_res = run_wilcoxon_test(ranking1, ranking2)
        cohen_d = calc_cohen(ranking1, ranking2)
        cliff_delta = compute_cliffs_delta(ranking1, ranking2)

        x, y = _align_ranking_scores(ranking1, ranking2)
        diffs = (np.array(x) - np.array(y)).tolist()

        ci_diff = (
            compute_bootstrap_ci(diffs)
            if len(diffs) > 0
            else ConfidenceIntervalResult(lower=0.0, upper=0.0, confidence_level=0.95)
        )

        s1 = compute_statistical_summary(x)
        s2 = compute_statistical_summary(y)

        def _quartiles(sorted_data: list[float]) -> tuple[float, float]:
            n = len(sorted_data)
            if n == 0:
                return 0.0, 0.0

            def _q(p: float) -> float:
                idx = p * (n - 1)
                lo = int(idx)
                hi = min(lo + 1, n - 1)
                return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])

            return _q(0.25), _q(0.75)

        q1_x, q3_x = _quartiles(sorted(x))
        sum1 = SummaryStatistics(
            mean=s1.mean,
            median=s1.median,
            variance=s1.variance,
            std_dev=s1.std_dev,
            min_val=min(x) if x else 0.0,
            max_val=max(x) if x else 0.0,
            q1=q1_x,
            q3=q3_x,
            count=s1.sample_size,
        )
        q1_y, q3_y = _quartiles(sorted(y))
        sum2 = SummaryStatistics(
            mean=s2.mean,
            median=s2.median,
            variance=s2.variance,
            std_dev=s2.std_dev,
            min_val=min(y) if y else 0.0,
            max_val=max(y) if y else 0.0,
            q1=q1_y,
            q3=q3_y,
            count=s2.sample_size,
        )

        return StatisticalReport(
            summary_statistics={"ranking1": sum1, "ranking2": sum2},
            correlations={"spearman": spearman, "kendall_tau": kendall},
            hypothesis_tests=[t_test, wilcoxon_res],
            effect_sizes=[cohen_d, cliff_delta],
            confidence_intervals={"differences": ci_diff},
            metadata={"sample_size": len(x)},
        )


def compute_statistical_summary(
    data: Sequence[float], n_bootstrap: int = 1000, seed: int = 42
) -> StatisticalSummary:
    """Compute comprehensive statistical metrics for a numeric sample."""
    n = len(data)
    if n == 0:
        return StatisticalSummary(
            mean=0.0,
            median=0.0,
            variance=0.0,
            std_dev=0.0,
            ci_95_lower=0.0,
            ci_95_upper=0.0,
            bootstrap_ci_95_lower=0.0,
            bootstrap_ci_95_upper=0.0,
            sample_size=0,
        )

    sorted_data = sorted(data)
    mean_val = sum(data) / n

    if n % 2 == 1:
        median_val = sorted_data[n // 2]
    else:
        median_val = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0

    var_val = sum((x - mean_val) ** 2 for x in data) / (n - 1 if n > 1 else 1)
    std_val = math.sqrt(var_val)

    se = std_val / math.sqrt(n) if n > 0 else 0.0
    ci_lower = mean_val - 1.96 * se
    ci_upper = mean_val + 1.96 * se

    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_bootstrap):
        resample = [rng.choice(sorted_data) for _ in range(n)]
        boot_means.append(sum(resample) / n)
    boot_means.sort()

    boot_lower_idx = int(0.025 * n_bootstrap)
    boot_upper_idx = int(0.975 * n_bootstrap)
    boot_lower = boot_means[boot_lower_idx]
    boot_upper = boot_means[boot_upper_idx]

    return StatisticalSummary(
        mean=round(mean_val, 4),
        median=round(median_val, 4),
        variance=round(var_val, 4),
        std_dev=round(std_val, 4),
        ci_95_lower=round(ci_lower, 4),
        ci_95_upper=round(ci_upper, 4),
        bootstrap_ci_95_lower=round(boot_lower, 4),
        bootstrap_ci_95_upper=round(boot_upper, 4),
        sample_size=n,
    )


def compute_cohens_d(group1: Sequence[float], group2: Sequence[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    m1 = sum(group1) / n1
    m2 = sum(group2) / n2
    v1 = sum((x - m1) ** 2 for x in group1) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in group2) / (n2 - 1)

    s_pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if s_pooled == 0.0:
        return 0.0
    return round((m1 - m2) / s_pooled, 4)


def perform_cross_validation_check(
    seed_results: dict[int, list[float]],
) -> dict[str, Any]:
    """Perform multi-seed cross-validation stability analysis across 5 seeds."""
    all_scores: list[float] = []
    seed_means: dict[int, float] = {}

    for seed, scores in seed_results.items():
        if scores:
            m = sum(scores) / len(scores)
            seed_means[seed] = round(m, 4)
            all_scores.extend(scores)

    summary = compute_statistical_summary(all_scores)
    means_list = list(seed_means.values())
    seed_var = compute_statistical_summary(means_list).variance if means_list else 0.0

    return {
        "overall_summary": summary.model_dump(),
        "seed_means": seed_means,
        "inter_seed_variance": round(seed_var, 6),
        "cross_validation_stability": "HIGH" if seed_var < 0.01 else "MODERATE",
    }
