from __future__ import annotations

from llm_reliability.records.metric import MetricRecord
from llm_reliability.statistics.auto_selection import (
    auto_select,
    run_recommended_test,
    suggest_correction,
)


def generate_reliability_statistical_report(
    metrics: list[MetricRecord],
    alpha: float = 0.05,
) -> dict:
    """Generate a statistical report from metric records using auto-selection.

    Groups records by benchmark, extracts composite_reliability scores,
    runs the auto-selected statistical test, and returns a structured report.
    """
    if not metrics:
        return {
            "n_metrics": 0,
            "n_groups": 0,
            "warning": "No metrics provided.",
        }

    groups: dict[str, list[float]] = {}
    for m in metrics:
        key = f"{m.agent}@{m.benchmark}"
        groups.setdefault(key, []).append(m.composite_reliability)

    if len(groups) < 2:
        return {
            "n_metrics": len(metrics),
            "n_groups": len(groups),
            "warning": "Need at least 2 groups for statistical comparison.",
            "groups": {
                k: {"count": len(v), "mean": float(__import__("numpy").mean(v))}
                for k, v in groups.items()
            },
        }

    samples = list(groups.values())
    group_labels = list(groups.keys())

    selection = auto_select(samples)
    test_result = run_recommended_test(samples, alpha=alpha)
    n_comparisons = len(groups)
    correction = suggest_correction(n_comparisons)

    return {
        "n_metrics": len(metrics),
        "n_groups": len(groups),
        "group_labels": group_labels,
        "group_stats": {
            k: {
                "count": len(v),
                "mean": float(__import__("numpy").mean(v)),
                "std": float(__import__("numpy").std(v, ddof=1)) if len(v) > 1 else 0.0,
            }
            for k, v in groups.items()
        },
        "recommended_test": selection.get("recommended_test"),
        "rationale": "",
        "test_result": test_result,
        "recommended_correction": correction,
        "alpha": alpha,
    }
