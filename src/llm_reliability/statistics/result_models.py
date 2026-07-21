"""
Pydantic v2 models for the Statistical Analysis Engine.

Defines the structure for all statistical outputs, ensuring strict typing
and validation of statistical metrics.
"""

from typing import Any
from pydantic import BaseModel, Field, field_validator


class CorrelationResult(BaseModel):
    """Result of a correlation analysis (e.g., Spearman, Kendall)."""

    coefficient: float = Field(..., ge=-1.0, le=1.0, description="Correlation coefficient.")
    p_value: float = Field(..., ge=0.0, le=1.0, description="Two-tailed p-value.")
    method: str = Field(..., description="Method used (e.g., Spearman, Kendall).")


class HypothesisTestResult(BaseModel):
    """Result of a hypothesis test (e.g., Paired t-test, Wilcoxon)."""

    statistic: float = Field(..., description="Test statistic.")
    p_value: float = Field(..., ge=0.0, le=1.0, description="P-value of the test.")
    method: str = Field(..., description="Name of the test.")
    alternative: str = Field("two-sided", description="Alternative hypothesis.")
    assumptions_met: bool = Field(..., description="Whether the statistical assumptions were satisfied.")
    warnings: list[str] = Field(default_factory=list, description="Warnings regarding assumptions or sample sizes.")


class EffectSizeResult(BaseModel):
    """Result of an effect size calculation (e.g., Cohen's d, Cliff's Delta)."""

    value: float = Field(..., description="Calculated effect size value.")
    method: str = Field(..., description="Name of the effect size metric.")
    interpretation: str = Field(..., description="Qualitative interpretation (e.g., small, medium, large).")


class ConfidenceIntervalResult(BaseModel):
    """Confidence interval bounds (e.g., via bootstrapping)."""

    lower: float = Field(..., description="Lower bound of the confidence interval.")
    upper: float = Field(..., description="Upper bound of the confidence interval.")
    confidence_level: float = Field(0.95, ge=0.0, le=1.0, description="Confidence level (e.g., 0.95).")

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence_level(cls, val: float) -> float:
        if not (0.0 < val < 1.0):
            raise ValueError("confidence_level must be strictly between 0.0 and 1.0.")
        return val


class SummaryStatistics(BaseModel):
    """Summary statistics for a dataset."""

    mean: float = Field(..., description="Arithmetic mean.")
    median: float = Field(..., description="Median value.")
    variance: float = Field(..., ge=0.0, description="Unbiased sample variance.")
    std_dev: float = Field(..., ge=0.0, description="Sample standard deviation.")
    min_val: float = Field(..., description="Minimum value.")
    max_val: float = Field(..., description="Maximum value.")
    q1: float = Field(..., description="First quartile (25th percentile).")
    q3: float = Field(..., description="Third quartile (75th percentile).")
    count: int = Field(..., ge=0, description="Sample size.")


class StatisticalReport(BaseModel):
    """Comprehensive report containing all generated statistical outputs."""

    summary_statistics: dict[str, SummaryStatistics] = Field(
        ...,
        description="Summary statistics mapped by variable/group name."
    )
    correlations: dict[str, CorrelationResult] = Field(
        default_factory=dict,
        description="Correlation results mapped by analysis name."
    )
    hypothesis_tests: list[HypothesisTestResult] = Field(
        default_factory=list,
        description="Hypothesis test results."
    )
    effect_sizes: list[EffectSizeResult] = Field(
        default_factory=list,
        description="Effect size calculations."
    )
    confidence_intervals: dict[str, ConfidenceIntervalResult] = Field(
        default_factory=dict,
        description="Confidence intervals mapped by variable name."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context such as sample sizes and timestamps."
    )
