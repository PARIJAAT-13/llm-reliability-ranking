"""
Tests for RankingEngine.
"""

import pytest

from llm_reliability.ranking.ranking_engine import RankingEngine
from llm_reliability.ranking.success_ranker import SuccessRanker
from llm_reliability.records.ranking import RankingRecord
from tests.ranking_test_helpers import create_mock_metric


def test_ranking_engine_orchestration():
    metrics = [
        create_mock_metric("agent_a", success_rate=0.8, composite=0.9),
        create_mock_metric("agent_b", success_rate=0.7, composite=0.95),
    ]
    engine = RankingEngine(metrics)

    # Success ranking
    ranking_success = engine.rank_success("2026-07-21T02:00:00Z")
    assert ranking_success.ranking_type == "success"
    assert ranking_success.rankings[0][0] == "agent_a"

    # Reliability ranking
    ranking_rel = engine.rank_reliability("2026-07-21T02:00:00Z")
    assert ranking_rel.ranking_type == "reliability"
    assert ranking_rel.rankings[0][0] == "agent_b"

    # Weighted ranking
    weights = {"success_rate": 0.5, "composite_reliability": 0.5}
    ranking_weighted = engine.rank_weighted(weights, "2026-07-21T02:00:00Z")
    assert ranking_weighted.ranking_type == "weighted"


def test_ranking_engine_validation_empty():
    engine = RankingEngine([])
    with pytest.raises(ValueError, match="Metrics list cannot be empty"):
        engine.validate()


def test_ranking_engine_validation_duplicate_agents():
    metrics = [
        create_mock_metric("agent_a"),
        create_mock_metric("agent_a"),
    ]
    engine = RankingEngine(metrics)
    with pytest.raises(ValueError, match="Duplicate agent found"):
        engine.validate()


def test_ranking_engine_validation_benchmark_mismatch():
    metrics = [
        create_mock_metric("agent_a", benchmark="bench-1"),
        create_mock_metric("agent_b", benchmark="bench-2"),
    ]
    engine = RankingEngine(metrics)
    with pytest.raises(ValueError, match="Benchmark mismatch"):
        engine.validate()


def test_ranking_engine_generate_with_custom_strategy():
    metrics = [
        create_mock_metric("agent_a", success_rate=0.8),
        create_mock_metric("agent_b", success_rate=0.9),
    ]
    engine = RankingEngine(metrics)
    ranking = engine.generate(SuccessRanker(), "2026-07-21T02:00:00Z")
    assert ranking.ranking_type == "success"
    assert ranking.rankings[0][0] == "agent_b"


def test_ranking_engine_sort():
    metrics = [
        create_mock_metric("agent_a", success_rate=0.8),
        create_mock_metric("agent_b", success_rate=0.9),
    ]
    engine = RankingEngine(metrics)
    sorted_tuples = engine.sort(lambda m: m.success_rate)
    assert sorted_tuples == (
        ("agent_b", 0.9),
        ("agent_a", 0.8),
    )


def test_ranking_engine_stateless_invocation():
    # Calling methods by passing metrics parameter directly
    metrics = [
        create_mock_metric("agent_a", success_rate=0.8),
        create_mock_metric("agent_b", success_rate=0.9),
    ]
    engine = RankingEngine()  # Uninitialized engine
    
    with pytest.raises(ValueError, match="No metrics provided"):
        engine.rank_success("2026-07-21T02:00:00Z")

    ranking = engine.rank_success("2026-07-21T02:00:00Z", metrics=metrics)
    assert ranking.rankings[0][0] == "agent_b"


def test_ranking_record_serialization_roundtrip():
    metrics = [
        create_mock_metric("agent_a", success_rate=0.8),
        create_mock_metric("agent_b", success_rate=0.9),
    ]
    engine = RankingEngine(metrics)
    ranking = engine.rank_success("2026-07-21T02:00:00Z")
    
    json_str = ranking.canonical_json()
    loaded_ranking = RankingRecord.from_canonical_json(json_str)
    assert loaded_ranking.ranking_type == ranking.ranking_type
    assert loaded_ranking.benchmark == ranking.benchmark
    assert loaded_ranking.rankings == ranking.rankings
    assert loaded_ranking.rank_map == ranking.rank_map
    assert loaded_ranking.computed_at == ranking.computed_at
