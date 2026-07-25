"""
LLM Reliability - Reliability Module.

Provides repeated-run execution, prompt perturbation, fault injection, and
composite score calculation mechanisms to evaluate repeated-run consistency,
prompt perturbation robustness, and fault tolerance.
"""

from __future__ import annotations

from llm_reliability.reliability.faults import (
    FaultInjectionStrategy,
    FaultManager,
    FaultReport,
    FaultReportGenerator,
    FaultRunResult,
    FaultTrace,
)
from llm_reliability.reliability.perturbation import (
    PerturbationManager,
    PerturbationRunResult,
    PerturbationStrategy,
)
from llm_reliability.reliability.repeated_runner import (
    RepeatedRunner,
    RepeatedRunResult,
)
from llm_reliability.reliability.score_calculator import (
    ReliabilityScore,
    ReliabilityScoreCalculator,
    ReliabilityScoreReport,
    ReliabilityWeights,
)

__all__ = [
    "RepeatedRunner",
    "RepeatedRunResult",
    "PerturbationStrategy",
    "PerturbationRunResult",
    "PerturbationManager",
    "FaultInjectionStrategy",
    "FaultTrace",
    "FaultRunResult",
    "FaultManager",
    "FaultReport",
    "FaultReportGenerator",
    "ReliabilityScoreCalculator",
    "ReliabilityScoreReport",
    "ReliabilityScore",
    "ReliabilityWeights",
]
