from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import Field

from llm_reliability.utils.serialization import SerializableModel


class CostEfficiencyResult(SerializableModel):
    total_cost_usd: float = Field(ge=0.0)
    total_calls: int = Field(ge=0)
    successful_calls: int = Field(ge=0)
    cost_per_success: float | None = Field(default=None, ge=0.0)
    cost_efficiency_score: float = Field(ge=0.0, le=1.0)
    cost_weighted_reliability: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


def compute_cost_per_success(
    total_cost_usd: float,
    successful_count: int,
) -> float | None:
    if successful_count <= 0:
        return None
    return total_cost_usd / successful_count


def compute_cost_efficiency_score(
    total_cost_usd: float,
    total_calls: int,
    *,
    reference_cost_per_call: float | None = None,
) -> float:
    if total_calls <= 0 or total_cost_usd <= 0:
        return 1.0
    cost_per_call = total_cost_usd / total_calls
    if reference_cost_per_call is not None and reference_cost_per_call > 0:
        ratio = cost_per_call / reference_cost_per_call
    else:
        ratio = cost_per_call
    return float(max(0.0, min(1.0, 1.0 / (1.0 + ratio))))


def compute_cost_weighted_reliability(
    reliability_score: float,
    cost_efficiency_score: float,
    *,
    cost_weight: float = 0.3,
) -> float:
    w = max(0.0, min(1.0, cost_weight))
    return (1.0 - w) * reliability_score + w * cost_efficiency_score


def compute_cost_efficiency(
    total_cost_usd: float | Decimal,
    total_calls: int,
    successful_calls: int,
    reliability_score: float,
    *,
    reference_cost_per_call: float | None = None,
    cost_weight: float = 0.3,
) -> CostEfficiencyResult:
    cost = float(total_cost_usd)
    cps = compute_cost_per_success(cost, successful_calls)
    ce_score = compute_cost_efficiency_score(
        cost, total_calls, reference_cost_per_call=reference_cost_per_call
    )
    cwr = compute_cost_weighted_reliability(reliability_score, ce_score, cost_weight=cost_weight)
    return CostEfficiencyResult(
        total_cost_usd=cost,
        total_calls=total_calls,
        successful_calls=successful_calls,
        cost_per_success=cps,
        cost_efficiency_score=ce_score,
        cost_weighted_reliability=cwr,
    )
