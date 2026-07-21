"""
Tests for SuccessRanker.
"""

import pytest

from llm_reliability.ranking.success_ranker import SuccessRanker
from tests.ranking_test_helpers import create_mock_metric


def test_success_ranking_order():
    metrics = [
        create_mock_metric("agent_a", success_rate=0.7),
        create_mock_metric("agent_b", success_rate=0.9),
        create_mock_metric("agent_c", success_rate=0.8),
    ]
    ranker = SuccessRanker()
    ranking = ranker.rank(metrics, computed_at="2026-07-21T02:00:00Z")

    assert ranking.ranking_type == "success"
    # Order should be agent_b (0.9), agent_c (0.8), agent_a (0.7)
    assert ranking.rankings == (
        ("agent_b", 0.9),
        ("agent_c", 0.8),
        ("agent_a", 0.7),
    )
    assert ranking.rank_map == {
        "agent_b": 1,
        "agent_c": 2,
        "agent_a": 3,
    }


def test_success_ranking_ties():
    metrics = [
        create_mock_metric("agent_c", success_rate=0.8),
        create_mock_metric("agent_a", success_rate=0.8),
        create_mock_metric("agent_b", success_rate=0.9),
    ]
    ranker = SuccessRanker()
    ranking = ranker.rank(metrics, computed_at="2026-07-21T02:00:00Z")

    # Order should be agent_b (0.9), then agent_a (0.8) and agent_c (0.8) broken alphabetically
    assert ranking.rankings == (
        ("agent_b", 0.9),
        ("agent_a", 0.8),
        ("agent_c", 0.8),
    )


# Removed test_success_ranking_missing_score because MetricRecord success_rate is non-nullable.
