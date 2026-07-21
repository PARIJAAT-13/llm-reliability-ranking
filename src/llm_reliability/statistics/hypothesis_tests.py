"""
Hypothesis testing module.

Performs Paired t-tests and Wilcoxon Signed-Rank tests, validating assumptions beforehand.
"""

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from llm_reliability.records.ranking import RankingRecord
from llm_reliability.statistics.assumptions import check_normality
from llm_reliability.statistics.correlation import _align_ranking_scores
from llm_reliability.statistics.result_models import HypothesisTestResult


def run_paired_t_test(
    ranking1: RankingRecord,
    ranking2: RankingRecord,
) -> HypothesisTestResult:
    """Perform a Paired Student's t-test on aligned scores.

    Checks the normality assumption on differences.
    """
    x, y = _align_ranking_scores(ranking1, ranking2)
    diffs = np.array(x) - np.array(y)

    assumptions_met, warning = check_normality(diffs)
    warnings = [warning] if warning else []

    if len(x) < 2:
        return HypothesisTestResult(
            statistic=0.0,
            p_value=1.0,
            method="Paired t-test",
            alternative="two-sided",
            assumptions_met=False,
            warnings=["Sample size too small to perform t-test (n < 2)."] + warnings,
        )

    # Perform t-test
    res = ttest_rel(x, y)
    
    # Handle NaN or constant arrays where std of diffs is 0
    import math
    stat = float(res.statistic) if not math.isnan(res.statistic) else 0.0
    p_val = float(res.pvalue) if not math.isnan(res.pvalue) else 1.0

    return HypothesisTestResult(
        statistic=stat,
        p_value=p_val,
        method="Paired t-test",
        alternative="two-sided",
        assumptions_met=assumptions_met,
        warnings=warnings,
    )


def run_wilcoxon_test(
    ranking1: RankingRecord,
    ranking2: RankingRecord,
) -> HypothesisTestResult:
    """Perform a Wilcoxon Signed-Rank test on aligned scores."""
    x, y = _align_ranking_scores(ranking1, ranking2)
    diffs = np.array(x) - np.array(y)

    warnings = []
    if len(x) < 5:
        warnings.append(f"Sample size too small for Wilcoxon signed-rank test (n={len(x)} < 5).")

    # Wilcoxon signed-rank test requires at least some non-zero differences
    if np.all(diffs == 0.0):
        return HypothesisTestResult(
            statistic=0.0,
            p_value=1.0,
            method="Wilcoxon Signed-Rank Test",
            alternative="two-sided",
            assumptions_met=True,
            warnings=["All differences are zero. Wilcoxon test cannot be computed."] + warnings,
        )

    try:
        res = wilcoxon(x, y)
        import math
        stat = float(res.statistic) if not math.isnan(res.statistic) else 0.0
        p_val = float(res.pvalue) if not math.isnan(res.pvalue) else 1.0
    except ValueError as e:
        # e.g., zero differences or other scipy constraints
        stat, p_val = 0.0, 1.0
        warnings.append(f"Wilcoxon test failed: {str(e)}")

    return HypothesisTestResult(
        statistic=stat,
        p_value=p_val,
        method="Wilcoxon Signed-Rank Test",
        alternative="two-sided",
        assumptions_met=True,
        warnings=warnings,
    )
