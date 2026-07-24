"""Tests for WeightedRankingConfig model."""

import pytest
from pydantic import ValidationError

from llm_reliability.ranking.ranking_models import WeightedRankingConfig


class TestWeightedRankingConfig:
    def test_valid_weights(self) -> None:
        config = WeightedRankingConfig(
            weights={
                "success_rate": 0.25,
                "repeated_run_consistency": 0.25,
                "perturbation_robustness": 0.25,
                "fault_tolerance": 0.25,
            }
        )
        assert config.weights["success_rate"] == 0.25

    def test_single_weight_valid(self) -> None:
        config = WeightedRankingConfig(weights={"success_rate": 1.0})
        assert config.weights["success_rate"] == 1.0

    def test_invalid_key_raises(self) -> None:
        with pytest.raises(ValidationError, match="Invalid weight key"):
            WeightedRankingConfig(weights={"invalid_key": 1.0})

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValidationError, match="cannot be negative"):
            WeightedRankingConfig(weights={"success_rate": -0.1, "repeated_run_consistency": 1.1})

    def test_weights_not_sum_to_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            WeightedRankingConfig(weights={"success_rate": 0.5, "repeated_run_consistency": 0.3})

    def test_composite_reliability_valid_key(self) -> None:
        config = WeightedRankingConfig(weights={"composite_reliability": 1.0})
        assert config.weights["composite_reliability"] == 1.0

    def test_all_valid_keys_accepted(self) -> None:
        weights = {
            "success_rate": 0.2,
            "repeated_run_consistency": 0.2,
            "perturbation_robustness": 0.2,
            "fault_tolerance": 0.2,
            "composite_reliability": 0.2,
        }
        config = WeightedRankingConfig(weights=weights)
        assert sum(config.weights.values()) == pytest.approx(1.0)
