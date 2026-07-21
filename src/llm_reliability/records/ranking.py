"""
Purpose
-------
Produce ordered agent rankings from aggregated metric records.

Responsibilities
----------------
- Rank agents by success rate or composite reliability
- Provide deterministic tie-breaking via agent name lexicographic order
- Support canonical serialization for downstream statistical analysis

Usage example
-------------
>>> from llm_reliability.records import RankingRecord, MetricRecord
>>> ranking = RankingRecord.from_metrics(metrics, ranking_type="reliability", computed_at="...")

Design notes
------------
RankingRecord derives exclusively from MetricRecord instances. Rank 1 is the
best score. Ties are broken lexicographically by agent name to guarantee
deterministic ordering across runs and platforms.
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from pydantic import Field

from llm_reliability.records.metric import MetricRecord
from llm_reliability.utils.serialization import SerializableModel

RankingType = Literal["success", "reliability", "weighted"]


RankingRecordT = TypeVar("RankingRecordT", bound="RankingRecord")


class RankingRecord(SerializableModel):
    """Immutable ordered ranking of agents for one benchmark."""

    ranking_type: RankingType
    benchmark: str = Field(min_length=1)
    rankings: tuple[tuple[str, float], ...]
    rank_map: dict[str, int]
    computed_at: str = Field(min_length=1)

    @classmethod
    def from_metrics(
        cls: type[RankingRecordT],
        metrics: list[MetricRecord],
        *,
        ranking_type: RankingType,
        computed_at: str,
    ) -> RankingRecordT:
        """Derive a ranking exclusively from metric records."""
        if not metrics:
            msg = "metrics must contain at least one MetricRecord"
            raise ValueError(msg)

        benchmark = metrics[0].benchmark
        if any(item.benchmark != benchmark for item in metrics):
            msg = "all metrics must share the same benchmark"
            raise ValueError(msg)

        if any(item.task_id is not None for item in metrics):
            msg = "rankings require benchmark-level MetricRecords (task_id must be None)"
            raise ValueError(msg)

        score_fn = (
            (lambda item: item.success_rate)
            if ranking_type == "success"
            else (lambda item: item.composite_reliability)
        )

        sorted_metrics = sorted(
            metrics,
            key=lambda item: (-score_fn(item), item.agent),
        )
        rankings = tuple((item.agent, score_fn(item)) for item in sorted_metrics)
        rank_map = {agent: index + 1 for index, (agent, _) in enumerate(rankings)}

        return cls(
            ranking_type=ranking_type,
            benchmark=benchmark,
            rankings=rankings,
            rank_map=rank_map,
            computed_at=computed_at,
        )
