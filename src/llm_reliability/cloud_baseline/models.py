from __future__ import annotations

import statistics
from decimal import Decimal

import numpy as np
from pydantic import Field

from llm_reliability.utils.serialization import SerializableModel


class CloudBaselineResult(SerializableModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    benchmark: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    success: bool
    score: float = Field(ge=0.0, le=1.0)
    cost_usd: Decimal = Field(default=Decimal("0"), ge=0)
    latency_ms: float = Field(ge=0.0)
    tokens_input: int = Field(ge=0)
    tokens_output: int = Field(ge=0)
    runtime_seconds: float = Field(ge=0.0)
    error: str | None = None


class CloudBaselineSummary(SerializableModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    benchmark: str = Field(min_length=1)
    total_cost_usd: Decimal = Field(default=Decimal("0"), ge=0)
    avg_latency_ms: float = Field(ge=0.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    p99_latency_ms: float = Field(ge=0.0)
    success_rate: float = Field(ge=0.0, le=1.0)
    total_tokens: int = Field(ge=0)
    task_count: int = Field(ge=0)
    cost_per_task_usd: Decimal = Field(default=Decimal("0"), ge=0)
    cost_per_success_usd: Decimal = Field(default=Decimal("0"), ge=0)

    @classmethod
    def compute_all(cls, results: list[CloudBaselineResult]) -> list[CloudBaselineSummary]:
        if not results:
            return []

        groups: dict[tuple[str, str, str], list[CloudBaselineResult]] = {}
        for r in results:
            key = (r.provider, r.model, r.benchmark)
            groups.setdefault(key, []).append(r)

        summaries: list[CloudBaselineSummary] = []
        for (provider, model, benchmark), group in groups.items():
            latencies = [r.latency_ms for r in group]
            successes = sum(1 for r in group if r.success)
            total_cost = sum((r.cost_usd for r in group), Decimal("0"))
            total_tok = sum(r.tokens_input + r.tokens_output for r in group)
            n = len(group)

            sorted_lat = sorted(latencies)
            p50 = _percentile(sorted_lat, 50)
            p95 = _percentile(sorted_lat, 95)
            p99 = _percentile(sorted_lat, 99)
            avg_lat = statistics.mean(latencies) if latencies else 0.0
            sr = successes / n if n > 0 else 0.0
            cpt = total_cost / Decimal(str(n)) if n > 0 else Decimal("0")
            cps = total_cost / Decimal(str(successes)) if successes > 0 else Decimal("1e10")

            summaries.append(
                cls(
                    provider=provider,
                    model=model,
                    benchmark=benchmark,
                    total_cost_usd=total_cost,
                    avg_latency_ms=avg_lat,
                    p50_latency_ms=p50,
                    p95_latency_ms=p95,
                    p99_latency_ms=p99,
                    success_rate=sr,
                    total_tokens=total_tok,
                    task_count=n,
                    cost_per_task_usd=cpt,
                    cost_per_success_usd=cps,
                )
            )
        return summaries


class CloudBaselineComparison(SerializableModel):
    benchmark: str = Field(min_length=1)
    summaries: list[CloudBaselineSummary] = Field(min_length=1)
    best_provider: str | None = None
    best_model: str | None = None
    best_score: float = Field(ge=0.0, le=1.0)
    most_efficient_provider: str | None = None
    most_efficient_model: str | None = None
    lowest_cost_per_success: Decimal = Field(default=Decimal("0"), ge=0)

    @classmethod
    def compute(cls, summaries: list[CloudBaselineSummary]) -> CloudBaselineComparison | None:
        if not summaries:
            return None
        benchmark = summaries[0].benchmark
        best = max(summaries, key=lambda s: s.success_rate)
        best_cost = min(
            (s for s in summaries if s.cost_per_success_usd != Decimal("inf")),
            key=lambda s: s.cost_per_success_usd,
            default=None,
        )
        return cls(
            benchmark=benchmark,
            summaries=summaries,
            best_provider=best.provider,
            best_model=best.model,
            best_score=best.success_rate,
            most_efficient_provider=best_cost.provider if best_cost else None,
            most_efficient_model=best_cost.model if best_cost else None,
            lowest_cost_per_success=best_cost.cost_per_success_usd if best_cost else Decimal("0"),
        )


def _percentile(sorted_data: list[float], pct: int) -> float:
    if not sorted_data:
        return 0.0
    return float(np.percentile(sorted_data, pct))
