"""
Ranking Strategy Interface.

Defines the abstract base class for all ranking strategies.
"""

from abc import ABC, abstractmethod

from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord


class RankingStrategy(ABC):
    """Abstract base class for ranking strategies."""

    @abstractmethod
    def rank(self, metrics: list[MetricRecord], computed_at: str) -> RankingRecord:
        """Generate a RankingRecord from a list of MetricRecords.

        Parameters
        ----------
        metrics : list[MetricRecord]
            The list of metric records for all agents to rank.
        computed_at : str
            ISO-8601 UTC timestamp of when this ranking was generated.

        Returns
        -------
        RankingRecord
            The generated ordered ranking record.
        """
        pass
