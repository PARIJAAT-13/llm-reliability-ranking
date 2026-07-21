"""
Success Ranking Strategy.

Ranks agents based on their success rate.
"""

from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord
from llm_reliability.ranking.ranking_strategy import RankingStrategy
from llm_reliability.ranking.utils import validate_metrics, sort_and_rank


class SuccessRanker(RankingStrategy):
    """Ranks agents by success_rate descending."""

    def rank(self, metrics: list[MetricRecord], computed_at: str) -> RankingRecord:
        """Generate a RankingRecord based on success_rate.

        Parameters
        ----------
        metrics : list[MetricRecord]
            List of metrics to rank.
        computed_at : str
            ISO-8601 UTC timestamp of generation.

        Returns
        -------
        RankingRecord
            Success-rate based rankings.
        """
        validate_metrics(metrics)
        
        # Ensure success_rate is not missing/None
        for m in metrics:
            if m.success_rate is None:
                raise ValueError(f"Agent '{m.agent}' has missing success_rate.")

        # Determine rankings and rank map
        rankings = sort_and_rank(metrics, lambda m: m.success_rate)
        rank_map = {agent: index + 1 for index, (agent, _) in enumerate(rankings)}

        return RankingRecord(
            ranking_type="success",
            benchmark=metrics[0].benchmark,
            rankings=rankings,
            rank_map=rank_map,
            computed_at=computed_at,
        )
