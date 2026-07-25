"""
Composite reliability score computation.

Formula
-------
The composite score is a weighted average of the individual metrics.
By default, equal weights are applied across all available metrics.

    composite = Σ (w_i * m_i)   subject to Σ w_i = 1.0

Where m_i ∈ {success_rate, consistency, robustness, fault_tolerance}
and only non-None metrics are included.

If a caller provides custom weights they must:
  • Cover every non-None metric key.
  • Be non-negative.
  • Have a total > 0.

Weights are normalised internally so they sum to 1.0 before computing
the composite.  This allows callers to pass e.g. {"success_rate": 2,
"consistency": 1} and have it treated as (⅔, ⅓).

Intuition: the composite score provides a single sortable number for
leaderboard rankings while remaining transparent about its constituent
parts.  Custom weights allow researchers to emphasise the metric most
relevant to their deployment context.
"""

from __future__ import annotations

import numpy as np


def compute_composite(
    success_rate: float,
    consistency: float,
    robustness: float | None = None,
    fault_tolerance: float | None = None,
    isr_composite: float | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute the composite reliability score.

    Args:
        success_rate:    Success rate metric in [0, 1].
        consistency:     Consistency metric in [0, 1].
        robustness:      Optional robustness metric in [0, 1].
        fault_tolerance: Optional fault tolerance metric in [0, 1].
        weights:         Optional custom weight mapping.  Keys must be drawn
                         from {'success_rate', 'consistency', 'robustness',
                         'fault_tolerance'} and must cover every non-None
                         metric.  Values must be non-negative and sum > 0.
                         They will be normalised to sum to 1.0.

    Returns:
        A tuple of (composite_score, effective_weights) where
        effective_weights is the normalised weight dict that was used.

    Raises:
        ValueError: If weights reference unknown metrics, miss available
                    metrics, contain negative values, or have zero total.
    """
    available: dict[str, float] = {
        "success_rate": success_rate,
        "consistency": consistency,
    }
    if robustness is not None:
        available["robustness"] = robustness
    if fault_tolerance is not None:
        available["fault_tolerance"] = fault_tolerance
    if isr_composite is not None:
        available["isr_composite"] = isr_composite

    if weights is None:
        n = len(available)
        effective_weights = {k: 1.0 / n for k in available}
    else:
        unknown = set(weights) - set(available)
        if unknown:
            raise ValueError(
                f"Unknown metric keys in weights: {unknown}. Available metrics: {set(available)}."
            )
        missing = set(available) - set(weights)
        if missing:
            raise ValueError(f"Weights missing for available metrics: {missing}.")
        for key, val in weights.items():
            if val < 0:
                raise ValueError(f"Weight for '{key}' cannot be negative: {val}.")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError(f"Weights must sum to a positive value, got {total:.6f}.")
        effective_weights = {k: v / total for k, v in weights.items()}

    composite = float(np.sum([effective_weights[k] * v for k, v in available.items()]))
    composite = float(np.clip(composite, 0.0, 1.0))
    return composite, effective_weights
