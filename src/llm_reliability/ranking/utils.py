"""
Utility functions for the Ranking Engine.

Includes validation, deterministic tie-breaking sorting, and other helper functions.
"""

from __future__ import annotations

from collections.abc import Callable

from llm_reliability.records.metric import MetricRecord


def validate_metrics(metrics: list[MetricRecord]) -> None:
    """Validate a list of MetricRecords.

    Raises ValueError if:
    - The list is empty.
    - There are duplicate agents in the list.
    - Any MetricRecord does not have the same benchmark.
    - Any MetricRecord has task_id is not None (rankings require benchmark-level metrics).
    """
    if not metrics:
        raise ValueError("Metrics list cannot be empty.")

    benchmark = metrics[0].benchmark
    seen_agents = set()
    for metric in metrics:
        if metric.benchmark != benchmark:
            raise ValueError(
                f"Benchmark mismatch: expected '{benchmark}', got '{metric.benchmark}'."
            )
        if metric.task_id is not None:
            raise ValueError(
                "Rankings require benchmark-level MetricRecords (task_id must be None)."
            )
        if metric.agent in seen_agents:
            raise ValueError(f"Duplicate agent found: '{metric.agent}'.")
        seen_agents.add(metric.agent)


def sort_and_rank(
    metrics: list[MetricRecord],
    score_extractor: Callable,
) -> tuple[tuple[str, float], ...]:
    """Sort agents by score descending, breaking ties lexicographically by agent name.

    Tie-breaking Algorithm:
    -----------------------
    1. Primary sort key: Score descending (higher score is better).
    2. Secondary sort key (tie-breaker): Agent name lexicographically ascending (alphabetical).
       Since agent names are unique in a valid validation list, this guarantees
       a deterministic, platform-independent ordering.
    """
    # Sort key: (-score, agent_name)
    sorted_metrics = sorted(
        metrics,
        key=lambda m: (-score_extractor(m), m.agent),
    )
    return tuple((m.agent, score_extractor(m)) for m in sorted_metrics)
