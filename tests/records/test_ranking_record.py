"""Tests for RankingRecord (Artifact 7)."""

import pytest
from pydantic import ValidationError

from llm_reliability.records import EvaluationRecord, ExecutionRecord, MetricRecord, RankingRecord
from tests.conftest import CONFIG_HASH, TIMESTAMP


def _make_execution(agent: str) -> ExecutionRecord:
    return ExecutionRecord(
        configuration_hash=CONFIG_HASH,
        seed=42,
        benchmark="mock",
        agent=agent,
        task_id="task-1",
        run_index=0,
        runtime_seconds=1.0,
        timestamp=TIMESTAMP,
        stdout="ok",
        stderr="",
        status="success",
    )


def _make_metric(agent: str, *, success: bool) -> MetricRecord:
    evaluation = EvaluationRecord.from_execution(
        _make_execution(agent),
        success=success,
        score=1.0 if success else 0.0,
        evaluated_at=TIMESTAMP,
    )
    return MetricRecord.from_evaluations([evaluation], computed_at=TIMESTAMP)


def test_success_ranking_orders_by_success_rate() -> None:
    metrics = [
        _make_metric("agent_b", success=True),
        _make_metric("agent_a", success=False),
    ]
    ranking = RankingRecord.from_metrics(
        metrics,
        ranking_type="success",
        computed_at=TIMESTAMP,
    )
    assert ranking.rankings[0][0] == "agent_b"
    assert ranking.rank_map["agent_b"] == 1
    assert ranking.rank_map["agent_a"] == 2


def test_reliability_ranking_type() -> None:
    metrics = [
        _make_metric("agent_a", success=True),
        _make_metric("agent_b", success=True),
    ]
    ranking = RankingRecord.from_metrics(
        metrics,
        ranking_type="reliability",
        computed_at=TIMESTAMP,
    )
    assert ranking.ranking_type == "reliability"


def test_tie_breaks_lexicographically() -> None:
    metrics = [
        _make_metric("agent_b", success=True),
        _make_metric("agent_a", success=True),
    ]
    ranking = RankingRecord.from_metrics(
        metrics,
        ranking_type="success",
        computed_at=TIMESTAMP,
    )
    assert ranking.rankings[0][0] == "agent_a"


def test_rejects_empty_metrics() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RankingRecord.from_metrics([], ranking_type="success", computed_at=TIMESTAMP)


def test_rejects_task_level_metrics() -> None:
    evaluation = EvaluationRecord.from_execution(
        _make_execution("agent_a"),
        success=True,
        score=1.0,
        evaluated_at=TIMESTAMP,
    )
    metric = MetricRecord.from_evaluations(
        [evaluation],
        task_id="task-1",
        computed_at=TIMESTAMP,
    )
    with pytest.raises(ValueError, match="task_id must be None"):
        RankingRecord.from_metrics([metric], ranking_type="success", computed_at=TIMESTAMP)


def test_serialization_round_trip() -> None:
    metrics = [_make_metric("agent_a", success=True)]
    ranking = RankingRecord.from_metrics(
        metrics,
        ranking_type="success",
        computed_at=TIMESTAMP,
    )
    restored = RankingRecord.from_canonical_json(ranking.canonical_json())
    assert ranking == restored


def test_immutable() -> None:
    metrics = [_make_metric("agent_a", success=True)]
    ranking = RankingRecord.from_metrics(
        metrics,
        ranking_type="success",
        computed_at=TIMESTAMP,
    )
    with pytest.raises(ValidationError):
        ranking.ranking_type = "reliability"  # type: ignore[misc]
