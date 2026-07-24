"""
Weighted Ranking Strategy.

Ranks agents based on a user-defined weighted combination of metrics.
"""

from llm_reliability.ranking.ranking_strategy import RankingStrategy
from llm_reliability.ranking.utils import sort_and_rank, validate_metrics
from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord


class WeightedRanker(RankingStrategy):
    """Ranks agents by a custom weighted combination of metric fields."""

    VALID_KEYS = {
        "success_rate",
        "repeated_run_consistency",
        "perturbation_robustness",
        "fault_tolerance",
        "composite_reliability",
    }

    def __init__(self, weights: dict[str, float]) -> None:
        """Initialize the WeightedRanker.

        Parameters
        ----------
        weights : dict[str, float]
            A dictionary mapping metric names to their weights.
            Weights must sum to 1.0.
        """
        self.weights = weights
        self.validate_weights()

    def validate_weights(self) -> None:
        """Validate the weights dictionary.

        Raises ValueError if:
        - Weights dictionary is empty.
        - Any key is invalid.
        - Any weight value is negative.
        - Weights do not sum to 1.0 (within 1e-6 tolerance).
        """
        if not self.weights:
            raise ValueError("Weights dictionary cannot be empty.")

        for key, val in self.weights.items():
            if key not in self.VALID_KEYS:
                raise ValueError(f"Invalid weight key: '{key}'. Valid keys are: {self.VALID_KEYS}")
            if val < 0.0:
                raise ValueError(f"Weight for '{key}' cannot be negative: {val}")

        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}.")

    def _compute_weighted_score(self, m: MetricRecord) -> float:
        """Compute the weighted score for a MetricRecord."""
        score = 0.0
        for key, weight in self.weights.items():
            val = getattr(m, key)
            if val is None:
                # If weight > 0 but the metric score is missing/None, reject it
                if weight > 0.0:
                    raise ValueError(
                        f"Agent '{m.agent}' has missing score for weighted metric '{key}'."
                    )
                val = 0.0
            score += weight * val
        return score

    def rank(self, metrics: list[MetricRecord], computed_at: str) -> RankingRecord:
        """Generate a RankingRecord based on custom weights.

        Parameters
        ----------
        metrics : list[MetricRecord]
            List of metrics to rank.
        computed_at : str
            ISO-8601 UTC timestamp of generation.

        Returns
        -------
        RankingRecord
            Weighted rankings.
        """
        validate_metrics(metrics)

        # Determine rankings and rank map
        rankings = sort_and_rank(metrics, self._compute_weighted_score)
        rank_map = {agent: index + 1 for index, (agent, _) in enumerate(rankings)}

        return RankingRecord(
            ranking_type="weighted",
            benchmark=metrics[0].benchmark,
            rankings=rankings,
            rank_map=rank_map,
            computed_at=computed_at,
        )
