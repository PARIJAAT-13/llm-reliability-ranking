"""
Correlation analysis module.

Computes Kendall's Tau and Spearman Rank Correlation between two rankings.
"""

from __future__ import annotations

from scipy.stats import kendalltau, spearmanr

from llm_reliability.records.ranking import RankingRecord
from llm_reliability.statistics.result_models import CorrelationResult
from llm_reliability.statistics.utils import validate_rankings


def _align_ranking_scores(
    ranking1: RankingRecord,
    ranking2: RankingRecord,
) -> tuple[list[float], list[float]]:
    """Align the scores of two rankings by agent name.

    Returns two lists of scores, aligned such that index i in both lists
    represents the same agent.
    """
    validate_rankings(ranking1, ranking2)

    # Align by sorting agents alphabetically
    r1_dict = dict(ranking1.rankings)
    r2_dict = dict(ranking2.rankings)

    sorted_agents = sorted(r1_dict.keys())
    x = [r1_dict[agent] for agent in sorted_agents]
    y = [r2_dict[agent] for agent in sorted_agents]

    return x, y


def compute_kendall_tau(
    ranking1: RankingRecord,
    ranking2: RankingRecord,
) -> CorrelationResult:
    """Compute Kendall's Tau correlation coefficient and p-value."""
    x, y = _align_ranking_scores(ranking1, ranking2)
    res = kendalltau(x, y)

    import math

    coefficient = float(res.statistic) if not math.isnan(res.statistic) else 0.0
    p_value = float(res.pvalue) if not math.isnan(res.pvalue) else 1.0

    if len(x) < 2:
        coefficient, p_value = 1.0, 1.0

    return CorrelationResult(
        coefficient=coefficient,
        p_value=p_value,
        method="Kendall's Tau",
    )


def compute_spearman(
    ranking1: RankingRecord,
    ranking2: RankingRecord,
) -> CorrelationResult:
    """Compute Spearman's rank correlation coefficient and p-value."""
    x, y = _align_ranking_scores(ranking1, ranking2)
    res = spearmanr(x, y)

    import math

    coefficient = float(res.statistic) if not math.isnan(res.statistic) else 0.0
    p_value = float(res.pvalue) if not math.isnan(res.pvalue) else 1.0

    if len(x) < 2:
        coefficient, p_value = 1.0, 1.0

    return CorrelationResult(
        coefficient=coefficient,
        p_value=p_value,
        method="Spearman Rank Correlation",
    )
