"""
Tests for ReliabilityRanker.
"""

from llm_reliability.ranking.reliability_ranker import ReliabilityRanker
from tests.ranking_test_helpers import create_mock_metric


def test_reliability_ranking_order():
    metrics = [
        create_mock_metric("agent_a", composite=0.75),
        create_mock_metric("agent_b", composite=0.95),
        create_mock_metric("agent_c", composite=0.85),
    ]
    ranker = ReliabilityRanker()
    ranking = ranker.rank(metrics, computed_at="2026-07-21T02:00:00Z")

    assert ranking.ranking_type == "reliability"
    # Order should be agent_b (0.95), agent_c (0.85), agent_a (0.75)
    assert ranking.rankings == (
        ("agent_b", 0.95),
        ("agent_c", 0.85),
        ("agent_a", 0.75),
    )
    assert ranking.rank_map == {
        "agent_b": 1,
        "agent_c": 2,
        "agent_a": 3,
    }


def test_reliability_ranking_ties():
    metrics = [
        create_mock_metric("agent_c", composite=0.8),
        create_mock_metric("agent_a", composite=0.8),
        create_mock_metric("agent_b", composite=0.9),
    ]
    ranker = ReliabilityRanker()
    ranking = ranker.rank(metrics, computed_at="2026-07-21T02:00:00Z")

    # Order should be agent_b (0.9), then agent_a (0.8) and agent_c (0.8) broken alphabetically
    assert ranking.rankings == (
        ("agent_b", 0.9),
        ("agent_a", 0.8),
        ("agent_c", 0.8),
    )


# Removed test_reliability_ranking_missing_score because MetricRecord composite_reliability is non-nullable.
