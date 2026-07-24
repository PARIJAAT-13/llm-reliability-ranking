"""
Tests for WeightedRanker.
"""

import pytest

from llm_reliability.ranking.ranking_models import WeightedRankingConfig
from llm_reliability.ranking.weighted_ranker import WeightedRanker
from tests.ranking_test_helpers import create_mock_metric


def test_weighted_ranking_config_validation():
    # Valid config
    config = WeightedRankingConfig(weights={"success_rate": 0.5, "composite_reliability": 0.5})
    assert config.weights["success_rate"] == 0.5

    # Invalid sum
    with pytest.raises(ValueError):
        WeightedRankingConfig(weights={"success_rate": 0.5, "composite_reliability": 0.6})

    # Negative weight
    with pytest.raises(ValueError):
        WeightedRankingConfig(weights={"success_rate": -0.1, "composite_reliability": 1.1})

    # Invalid key
    with pytest.raises(ValueError):
        WeightedRankingConfig(weights={"invalid_metric": 1.0})


def test_weighted_ranking_order():
    metrics = [
        create_mock_metric("agent_a", success_rate=1.0, composite=0.0),
        create_mock_metric("agent_b", success_rate=0.0, composite=1.0),
        create_mock_metric("agent_c", success_rate=0.5, composite=0.5),
    ]

    # If we weight success_rate heavily (0.8) and composite lightly (0.2)
    # agent_a: 0.8 * 1.0 + 0.2 * 0.0 = 0.8
    # agent_b: 0.8 * 0.0 + 0.2 * 1.0 = 0.2
    # agent_c: 0.8 * 0.5 + 0.2 * 0.5 = 0.5
    # Order should be agent_a (0.8), agent_c (0.5), agent_b (0.2)
    weights = {"success_rate": 0.8, "composite_reliability": 0.2}
    ranker = WeightedRanker(weights)
    ranking = ranker.rank(metrics, computed_at="2026-07-21T02:00:00Z")

    assert ranking.ranking_type == "weighted"
    assert ranking.rankings == (
        ("agent_a", 0.8),
        ("agent_c", 0.5),
        ("agent_b", 0.2),
    )


def test_weighted_ranking_missing_score():
    # When a weight is non-zero, but the metric score is None
    metrics = [
        create_mock_metric("agent_a", success_rate=0.5, robustness=None),
    ]
    weights = {"success_rate": 0.5, "perturbation_robustness": 0.5}
    ranker = WeightedRanker(weights)

    with pytest.raises(ValueError, match="missing score for weighted metric"):
        ranker.rank(metrics, computed_at="2026-07-21T02:00:00Z")
