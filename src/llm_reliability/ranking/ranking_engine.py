"""
Ranking Engine.

Coordinates and executes agent rankings based on MetricRecords.
"""

from __future__ import annotations

from collections.abc import Callable

from llm_reliability.ranking.ranking_strategy import RankingStrategy
from llm_reliability.ranking.reliability_ranker import ReliabilityRanker
from llm_reliability.ranking.success_ranker import SuccessRanker
from llm_reliability.ranking.utils import sort_and_rank, validate_metrics
from llm_reliability.ranking.weighted_ranker import WeightedRanker
from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord


class RankingEngine:
    """Orchestrates agent rankings using various ranking strategies."""

    def __init__(self, metrics: list[MetricRecord] | None = None) -> None:
        """Initialize the RankingEngine with optional MetricRecords.

        Parameters
        ----------
        metrics : list[MetricRecord], optional
            Default list of metrics to rank.
        """
        self.metrics = metrics

    def validate(self, metrics: list[MetricRecord] | None = None) -> None:
        """Validate the list of MetricRecords.

        Parameters
        ----------
        metrics : list[MetricRecord], optional
            Override metrics list to validate. If None, validates self.metrics.
        """
        target = metrics if metrics is not None else self.metrics
        if target is None:
            raise ValueError("No metrics provided to validate.")
        validate_metrics(target)

    def sort(
        self,
        score_extractor: Callable[[MetricRecord], float],
        metrics: list[MetricRecord] | None = None,
    ) -> tuple[tuple[str, float], ...]:
        """Sort agents using a score extractor, breaking ties lexicographically.

        Parameters
        ----------
        score_extractor : Callable[[MetricRecord], float]
            Function extracting the score value from a MetricRecord.
        metrics : list[MetricRecord], optional
            Override metrics list.

        Returns
        -------
        tuple[tuple[str, float], ...]
            Ordered tuple of (agent_name, score).
        """
        target = metrics if metrics is not None else self.metrics
        if target is None:
            raise ValueError("No metrics provided to sort.")
        self.validate(target)
        return sort_and_rank(target, score_extractor)

    def rank_success(
        self,
        computed_at: str,
        metrics: list[MetricRecord] | None = None,
    ) -> RankingRecord:
        """Generate rankings based on success rate.

        Parameters
        ----------
        computed_at : str
            ISO-8601 UTC timestamp of generation.
        metrics : list[MetricRecord], optional
            Override metrics list.

        Returns
        -------
        RankingRecord
            Ordered success rate rankings.
        """
        target = metrics if metrics is not None else self.metrics
        if target is None:
            raise ValueError("No metrics provided to rank.")
        return SuccessRanker().rank(target, computed_at)

    def rank_reliability(
        self,
        computed_at: str,
        metrics: list[MetricRecord] | None = None,
    ) -> RankingRecord:
        """Generate rankings based on composite reliability.

        Parameters
        ----------
        computed_at : str
            ISO-8601 UTC timestamp of generation.
        metrics : list[MetricRecord], optional
            Override metrics list.

        Returns
        -------
        RankingRecord
            Ordered composite reliability rankings.
        """
        target = metrics if metrics is not None else self.metrics
        if target is None:
            raise ValueError("No metrics provided to rank.")
        return ReliabilityRanker().rank(target, computed_at)

    def rank_weighted(
        self,
        weights: dict[str, float],
        computed_at: str,
        metrics: list[MetricRecord] | None = None,
    ) -> RankingRecord:
        """Generate rankings based on custom weights.

        Parameters
        ----------
        weights : dict[str, float]
            Weight dictionary mapping metric fields to weights.
        computed_at : str
            ISO-8601 UTC timestamp of generation.
        metrics : list[MetricRecord], optional
            Override metrics list.

        Returns
        -------
        RankingRecord
            Ordered weighted rankings.
        """
        target = metrics if metrics is not None else self.metrics
        if target is None:
            raise ValueError("No metrics provided to rank.")
        return WeightedRanker(weights).rank(target, computed_at)

    def generate(
        self,
        strategy: RankingStrategy,
        computed_at: str,
        metrics: list[MetricRecord] | None = None,
    ) -> RankingRecord:
        """Generate rankings using an arbitrary custom strategy.

        Parameters
        ----------
        strategy : RankingStrategy
            The custom ranking strategy to execute.
        computed_at : str
            ISO-8601 UTC timestamp of generation.
        metrics : list[MetricRecord], optional
            Override metrics list.

        Returns
        -------
        RankingRecord
            Generated rankings.
        """
        target = metrics if metrics is not None else self.metrics
        if target is None:
            raise ValueError("No metrics provided to rank.")
        return strategy.rank(target, computed_at)
