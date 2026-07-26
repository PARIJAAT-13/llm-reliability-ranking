from __future__ import annotations

import warnings as _warnings

from llm_reliability.metrics.composite import compute_composite
from llm_reliability.metrics.consistency import compute_consistency
from llm_reliability.metrics.fault_tolerance import compute_fault_tolerance
from llm_reliability.metrics.models import ReliabilityResult
from llm_reliability.metrics.reliability_engine import ReliabilityEngine
from llm_reliability.metrics.robustness import compute_robustness
from llm_reliability.reliability.metrics.isr import compute_isr, compute_temporal_isr

_warnings.warn(
    "llm_reliability.metrics is deprecated. Use llm_reliability.reliability.metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ReliabilityEngine",
    "ReliabilityResult",
    "compute_consistency",
    "compute_robustness",
    "compute_fault_tolerance",
    "compute_isr",
    "compute_temporal_isr",
    "compute_composite",
]
