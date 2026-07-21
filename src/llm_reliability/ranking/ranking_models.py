"""
Pydantic models for the Ranking Engine.
"""

from pydantic import BaseModel, Field, field_validator


class WeightedRankingConfig(BaseModel):
    """Configuration schema for weighted rankings."""

    weights: dict[str, float] = Field(
        ...,
        description="Weights for each metric. Must sum to 1.0."
    )

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        valid_keys = {
            "success_rate",
            "repeated_run_consistency",
            "perturbation_robustness",
            "fault_tolerance",
            "composite_reliability",
        }
        for key, val in weights.items():
            if key not in valid_keys:
                raise ValueError(f"Invalid weight key: '{key}'. Valid keys are: {valid_keys}")
            if val < 0.0:
                raise ValueError(f"Weight for '{key}' cannot be negative: {val}")

        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}.")
        return weights
