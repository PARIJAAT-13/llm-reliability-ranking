"""
Purpose
-------
Compute ranking divergence and overlap metrics between two ``RankingRecord``
objects, capturing where Success Ranking and Reliability Ranking diverge.

Responsibilities
----------------
- ``compute_ranking_overlap``  — fraction of agent pairs in the same relative order
  in both rankings.
- ``compute_ranking_divergence`` — complement of overlap (1 − overlap).
- ``compute_rank_displacement`` — mean absolute rank-position change per agent.
- ``RankingDivergenceResult`` — Pydantic model packaging all three metrics.

Usage example
-------------
>>> from llm_reliability.statistics.ranking_divergence import analyze_ranking_divergence
>>> result = analyze_ranking_divergence(success_ranking, reliability_ranking)
>>> print(result.overlap)          # e.g. 0.6
>>> print(result.divergence)       # e.g. 0.4
>>> print(result.mean_displacement)

Design notes
------------
Overlap is computed via concordant-pair counting, equivalent to normalising
Kendall's Tau concordance count to [0, 1] without the sign.  A pair (i, j)
is *concordant* when the relative order of agents i and j is identical in
both rankings.

Mean rank displacement is the mean of |rank1[agent] − rank2[agent]|
over all agents present in both rankings.
"""

from __future__ import annotations

from itertools import combinations

from pydantic import BaseModel, Field

from llm_reliability.records.ranking import RankingRecord
from llm_reliability.statistics.utils import validate_rankings


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class RankingDivergenceResult(BaseModel):
    """Pairwise ranking divergence metrics.

    Attributes
    ----------
    overlap : float
        Fraction of agent pairs whose relative ordering is identical in both
        rankings. Range [0, 1]; 1.0 = perfect agreement.
    divergence : float
        Complement of overlap (1 − overlap). 0.0 = rankings are identical.
    mean_displacement : float
        Mean absolute rank-position shift per agent (summed over all agents
        and divided by the number of agents). 0.0 = identical ranks.
    max_displacement : int
        Largest single-agent absolute rank shift.
    n_agents : int
        Number of agents present in both rankings.
    n_concordant_pairs : int
        Number of concordant agent pairs.
    n_discordant_pairs : int
        Number of discordant agent pairs.
    ranking1_type : str
        ``ranking_type`` of the first RankingRecord.
    ranking2_type : str
        ``ranking_type`` of the second RankingRecord.
    benchmark : str
        Benchmark shared by both rankings.
    """

    overlap: float = Field(ge=0.0, le=1.0)
    divergence: float = Field(ge=0.0, le=1.0)
    mean_displacement: float = Field(ge=0.0)
    max_displacement: int = Field(ge=0)
    n_agents: int = Field(ge=0)
    n_concordant_pairs: int = Field(ge=0)
    n_discordant_pairs: int = Field(ge=0)
    ranking1_type: str
    ranking2_type: str
    benchmark: str


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------

def compute_ranking_overlap(
    ranking1: RankingRecord,
    ranking2: RankingRecord,
) -> float:
    """Compute the fraction of agent pairs in concordant relative order.

    A pair (a, b) is concordant when both rankings agree on which agent
    ranks higher.  Ties in score are counted as half-concordant.

    Parameters
    ----------
    ranking1 : RankingRecord
        First ranking.
    ranking2 : RankingRecord
        Second ranking.

    Returns
    -------
    float
        Overlap fraction in [0, 1].  Returns 1.0 when fewer than 2 agents.
    """
    validate_rankings(ranking1, ranking2)
    agents = sorted(set(dict(ranking1.rankings)) & set(dict(ranking2.rankings)))

    if len(agents) < 2:
        return 1.0

    r1 = dict(ranking1.rankings)
    r2 = dict(ranking2.rankings)

    concordant = 0.0
    total = 0

    for a, b in combinations(agents, 2):
        total += 1
        sign1 = _sign(r1[a] - r1[b])
        sign2 = _sign(r2[a] - r2[b])

        if sign1 == sign2:
            concordant += 1.0
        elif sign1 == 0 or sign2 == 0:
            concordant += 0.5  # tied in one ranking

    return concordant / total if total > 0 else 1.0


def compute_ranking_divergence(
    ranking1: RankingRecord,
    ranking2: RankingRecord,
) -> float:
    """Complement of ranking overlap: fraction of discordant agent pairs.

    Parameters
    ----------
    ranking1 : RankingRecord
        First ranking.
    ranking2 : RankingRecord
        Second ranking.

    Returns
    -------
    float
        Divergence in [0, 1].  0.0 = rankings are identical.
    """
    return 1.0 - compute_ranking_overlap(ranking1, ranking2)


def compute_rank_displacement(
    ranking1: RankingRecord,
    ranking2: RankingRecord,
) -> tuple[float, int]:
    """Compute mean and max absolute rank-position displacement per agent.

    Parameters
    ----------
    ranking1 : RankingRecord
        First ranking.
    ranking2 : RankingRecord
        Second ranking.

    Returns
    -------
    tuple[float, int]
        (mean_displacement, max_displacement).
        Both are 0 when fewer than 2 agents exist.
    """
    validate_rankings(ranking1, ranking2)
    shared_agents = sorted(
        set(ranking1.rank_map) & set(ranking2.rank_map)
    )

    if not shared_agents:
        return 0.0, 0

    displacements = [
        abs(ranking1.rank_map[a] - ranking2.rank_map[a]) for a in shared_agents
    ]
    return (
        sum(displacements) / len(displacements),
        max(displacements),
    )


# ---------------------------------------------------------------------------
# Convenience analyser
# ---------------------------------------------------------------------------

def analyze_ranking_divergence(
    ranking1: RankingRecord,
    ranking2: RankingRecord,
) -> RankingDivergenceResult:
    """Perform a full ranking divergence analysis.

    Parameters
    ----------
    ranking1 : RankingRecord
        First ranking (e.g. success ranking).
    ranking2 : RankingRecord
        Second ranking (e.g. reliability ranking).

    Returns
    -------
    RankingDivergenceResult
        Complete divergence metrics.
    """
    validate_rankings(ranking1, ranking2)

    overlap = compute_ranking_overlap(ranking1, ranking2)
    divergence = 1.0 - overlap
    mean_disp, max_disp = compute_rank_displacement(ranking1, ranking2)

    agents = sorted(set(dict(ranking1.rankings)) & set(dict(ranking2.rankings)))
    r1 = dict(ranking1.rankings)
    r2 = dict(ranking2.rankings)

    concordant = 0
    discordant = 0
    for a, b in combinations(agents, 2):
        sign1 = _sign(r1[a] - r1[b])
        sign2 = _sign(r2[a] - r2[b])
        if sign1 == sign2 and sign1 != 0:
            concordant += 1
        elif sign1 != 0 and sign2 != 0 and sign1 != sign2:
            discordant += 1

    return RankingDivergenceResult(
        overlap=round(overlap, 10),
        divergence=round(divergence, 10),
        mean_displacement=round(mean_disp, 6),
        max_displacement=max_disp,
        n_agents=len(agents),
        n_concordant_pairs=concordant,
        n_discordant_pairs=discordant,
        ranking1_type=ranking1.ranking_type,
        ranking2_type=ranking2.ranking_type,
        benchmark=ranking1.benchmark,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sign(x: float) -> int:
    """Return −1, 0, or +1."""
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0
