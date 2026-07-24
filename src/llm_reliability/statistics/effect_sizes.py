"""
Effect size computation module.

Computes Cohen's d, Rank-biserial correlation, and Cliff's Delta.
"""

import numpy as np
from scipy.stats import rankdata

from llm_reliability.records.ranking import RankingRecord
from llm_reliability.statistics.correlation import _align_ranking_scores
from llm_reliability.statistics.result_models import EffectSizeResult


def compute_cohens_d(
    ranking1: RankingRecord,
    ranking2: RankingRecord,
) -> EffectSizeResult:
    """Compute Cohen's d for the difference between two rankings."""
    x, y = _align_ranking_scores(ranking1, ranking2)

    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)

    pooled_std = np.sqrt((var_x + var_y) / 2.0)

    if pooled_std == 0.0:
        d = 0.0
    else:
        d = (mean_x - mean_y) / pooled_std

    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return EffectSizeResult(
        value=float(d),
        method="Cohen's d (pooled)",
        interpretation=interpretation,
    )


def compute_rank_biserial(
    ranking1: RankingRecord,
    ranking2: RankingRecord,
) -> EffectSizeResult:
    """Compute Rank-biserial correlation for paired rankings."""
    x, y = _align_ranking_scores(ranking1, ranking2)
    diffs = np.array(x) - np.array(y)

    # Exclude zero differences
    non_zero = diffs[diffs != 0.0]
    if len(non_zero) == 0:
        return EffectSizeResult(
            value=0.0,
            method="Rank-biserial Correlation",
            interpretation="negligible",
        )

    ranks = rankdata(np.abs(non_zero))
    signs = np.sign(non_zero)
    signed_ranks = ranks * signs

    r_plus = np.sum(signed_ranks[signed_ranks > 0])
    r_minus = np.sum(np.abs(signed_ranks[signed_ranks < 0]))

    total_ranks = r_plus + r_minus
    if total_ranks == 0.0:
        r = 0.0
    else:
        r = (r_plus - r_minus) / total_ranks

    abs_r = abs(r)
    if abs_r < 0.1:
        interpretation = "negligible"
    elif abs_r < 0.3:
        interpretation = "small"
    elif abs_r < 0.5:
        interpretation = "medium"
    else:
        interpretation = "large"

    return EffectSizeResult(
        value=float(r),
        method="Rank-biserial Correlation",
        interpretation=interpretation,
    )


def compute_cliffs_delta(
    ranking1: RankingRecord,
    ranking2: RankingRecord,
) -> EffectSizeResult:
    """Compute Cliff's Delta between two rankings."""
    x, y = _align_ranking_scores(ranking1, ranking2)
    n_x, n_y = len(x), len(y)

    greater = 0
    less = 0
    for val_x in x:
        for val_y in y:
            if val_x > val_y:
                greater += 1
            elif val_x < val_y:
                less += 1

    if n_x * n_y == 0:
        delta = 0.0
    else:
        delta = (greater - less) / (n_x * n_y)

    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "negligible"
    elif abs_delta < 0.33:
        interpretation = "small"
    elif abs_delta < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"

    return EffectSizeResult(
        value=float(delta),
        method="Cliff's Delta",
        interpretation=interpretation,
    )
