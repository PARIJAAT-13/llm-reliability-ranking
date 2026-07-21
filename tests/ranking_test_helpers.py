"""
Helpers for ranking tests.
"""

from llm_reliability.records.metric import MetricRecord


def create_mock_metric(
    agent: str,
    success_rate: float = 0.5,
    consistency: float = 0.5,
    robustness: float | None = None,
    fault_tolerance: float | None = None,
    composite: float = 0.5,
    benchmark: str = "mock-bench",
) -> MetricRecord:
    """Create a MetricRecord with default or custom fields for testing."""
    return MetricRecord(
        benchmark=benchmark,
        agent=agent,
        task_id=None,
        evaluation_count=10,
        success_rate=success_rate,
        repeated_run_consistency=consistency,
        perturbation_robustness=robustness,
        fault_tolerance=fault_tolerance,
        composite_reliability=composite,
        computed_at="2026-07-21T02:00:00Z",
    )
