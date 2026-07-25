"""
LLM Reliability - Fault Injection Framework.

Provides fault injection strategies, FaultManager orchestration, and FaultReport generation
for fault tolerance evaluation.
"""

from __future__ import annotations

from llm_reliability.reliability.faults.base import (
    FaultInjectionStrategy,
    FaultRunResult,
    FaultTrace,
    RecoveryStatus,
)
from llm_reliability.reliability.faults.extensions import (
    SEVERITY_PARAMS,
    FaultInjectionConfig,
    FaultSchedule,
    FaultSeverity,
    FaultSummary,
    RecoveryMetrics,
    ScheduledFaultConfig,
)
from llm_reliability.reliability.faults.manager import FaultManager
from llm_reliability.reliability.faults.report import (
    FaultReport,
    FaultReportGenerator,
    FaultTypeMetrics,
)
from llm_reliability.reliability.faults.strategies import (
    ArtificialTimeoutFaultStrategy,
    ContextTruncationFaultStrategy,
    InvalidModelResponseFaultStrategy,
    NetworkInterruptionFaultStrategy,
    TemporaryApiFailureFaultStrategy,
    ToolFailureFaultStrategy,
)

__all__ = [
    "FaultInjectionStrategy",
    "FaultTrace",
    "FaultRunResult",
    "RecoveryStatus",
    "FaultManager",
    "FaultReport",
    "FaultReportGenerator",
    "FaultTypeMetrics",
    "ArtificialTimeoutFaultStrategy",
    "TemporaryApiFailureFaultStrategy",
    "InvalidModelResponseFaultStrategy",
    "ToolFailureFaultStrategy",
    "ContextTruncationFaultStrategy",
    "NetworkInterruptionFaultStrategy",
    "FaultSeverity",
    "FaultSchedule",
    "ScheduledFaultConfig",
    "FaultInjectionConfig",
    "RecoveryMetrics",
    "FaultSummary",
    "SEVERITY_PARAMS",
]
