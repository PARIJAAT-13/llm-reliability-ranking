"""
LLM Reliability - Reliability Metrics Module.

Provides quantitative metric calculation across consistency, prompt perturbation robustness,
and fault tolerance dimensions.
"""

from llm_reliability.reliability.metrics.base import (
    ConsistencyMetricResult,
    FaultToleranceMetricResult,
    ReliabilityMetric,
    RobustnessMetricResult,
)
from llm_reliability.reliability.metrics.consistency import RepeatedRunConsistencyMetric
from llm_reliability.reliability.metrics.engine import (
    ReliabilityMetricsEngine,
    ScopeReliabilitySummary,
)
from llm_reliability.reliability.metrics.fault_tolerance import FaultToleranceMetric
from llm_reliability.reliability.metrics.report import (
    ReliabilityMetricReport,
    ReliabilityReportGenerator,
)
from llm_reliability.reliability.metrics.robustness import (
    PromptPerturbationRobustnessMetric,
)

__all__ = [
    "ReliabilityMetric",
    "ConsistencyMetricResult",
    "RobustnessMetricResult",
    "FaultToleranceMetricResult",
    "RepeatedRunConsistencyMetric",
    "PromptPerturbationRobustnessMetric",
    "FaultToleranceMetric",
    "ReliabilityMetricsEngine",
    "ScopeReliabilitySummary",
    "ReliabilityMetricReport",
    "ReliabilityReportGenerator",
]
