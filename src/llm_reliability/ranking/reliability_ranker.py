"""
Reliability Ranking Strategy.

Ranks agents based on their composite reliability.
"""

from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord
from llm_reliability.ranking.ranking_strategy import RankingStrategy
from llm_reliability.ranking.utils import validate_metrics, sort_and_rank


class ReliabilityRanker(RankingStrategy):
    """Ranks agents by composite_reliability descending."""

    def rank(self, metrics: list[MetricRecord], computed_at: str) -> RankingRecord:
        """Generate a RankingRecord based on composite_reliability.

        Parameters
        ----------
        metrics : list[MetricRecord]
            List of metrics to rank.
        computed_at : str
            ISO-8601 UTC timestamp of generation.

        Returns
        -------
        RankingRecord
            Reliability-based rankings.
        """
        validate_metrics(metrics)
        
        # Ensure composite_reliability is not missing/None
        for m in metrics:
            if m.composite_reliability is None:
                raise ValueError(f"Agent '{m.agent}' has missing composite_reliability.")

        # Determine rankings and rank map
        rankings = sort_and_rank(metrics, lambda m: m.composite_reliability)
        rank_map = {agent: index + 1 for index, (agent, _) in enumerate(rankings)}

        return RankingRecord(
            ranking_type="reliability",
            benchmark=metrics[0].benchmark,
            rankings=rankings,
            rank_map=rank_map,
            computed_at=computed_at,
        )
