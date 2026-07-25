"""
Extended fault injection capabilities: scheduling, severity, recovery metrics.

This module extends the existing fault injection framework with:

1. **FaultSeverity** — controls the intensity of injected faults
   (LIGHT, MODERATE, SEVERE, CRITICAL).  Each severity level changes
   how the fault strategy behaves (e.g., truncation ratio, delay duration).

2. **FaultSchedule** — controls WHEN faults are injected
   (FIRST_RUN, RANDOM_RUN, EVERY_RUN, SEQUENCE).

3. **RecoveryMetrics** — detailed analysis of agent recovery behaviour
   including recovery time, recovery success rate per severity, and
   degradation trajectory.

4. **FaultInjectionConfig** — a complete configuration model for designing
   fault injection experiments with probability, severity, and scheduling.

All classes are fully backward-compatible — they augment rather than
replace the existing FaultManager and FaultInjectionStrategy classes.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import Field

from llm_reliability.reliability.faults.base import FaultTrace
from llm_reliability.utils.serialization import SerializableModel

# ======================================================================
# 1. Fault Severity
# ======================================================================


class FaultSeverity(str, enum.Enum):
    """Controls the intensity of fault injection.

    Translates to concrete parameters for each strategy:
    - LIGHT:     minimal disruption (e.g., 10 % context truncation,
                 short delay, single failure)
    - MODERATE:  noticeable disruption (e.g., 30 % truncation,
                 moderate delay, a few failures)
    - SEVERE:    major disruption (e.g., 60 % truncation,
                 long delay, many failures)
    - CRITICAL:  near-total failure (e.g., 90 % truncation,
                 very long delay, permanent failure)
    """

    LIGHT = "light"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


# Mapping severity → parameter multipliers / overrides
SEVERITY_PARAMS: dict[FaultSeverity, dict[str, Any]] = {
    FaultSeverity.LIGHT: {
        "truncation_ratio": 0.1,
        "delay_seconds": 0.2,
        "max_failures": 1,
        "response_mode": "empty",
        "description": "Light disruption — minimal impact expected.",
    },
    FaultSeverity.MODERATE: {
        "truncation_ratio": 0.3,
        "delay_seconds": 1.0,
        "max_failures": 2,
        "response_mode": "malformed_json",
        "description": "Moderate disruption — performance degradation likely.",
    },
    FaultSeverity.SEVERE: {
        "truncation_ratio": 0.6,
        "delay_seconds": 3.0,
        "max_failures": 3,
        "response_mode": "unexpected_type",
        "description": "Severe disruption — significant performance drop expected.",
    },
    FaultSeverity.CRITICAL: {
        "truncation_ratio": 0.9,
        "delay_seconds": 10.0,
        "max_failures": 5,
        "response_mode": "empty",
        "description": "Critical disruption — agent is unlikely to recover.",
    },
}

# Which injection_point each severity is applicable to
SEVERITY_INJECTION_POINTS: dict[FaultSeverity, list[str]] = {
    FaultSeverity.LIGHT: ["prompt", "api_call", "agent_run", "tool_call"],
    FaultSeverity.MODERATE: ["prompt", "api_call", "agent_run", "tool_call"],
    FaultSeverity.SEVERE: ["prompt", "api_call", "agent_run"],
    FaultSeverity.CRITICAL: ["api_call", "agent_run"],
}


# ======================================================================
# 2. Fault Schedule
# ======================================================================


class FaultSchedule(str, enum.Enum):
    """Controls when faults are injected during repeated execution."""

    FIRST_RUN = "first_run"
    """Inject fault only on the first repetition."""

    LAST_RUN = "last_run"
    """Inject fault only on the last repetition."""

    RANDOM_RUN = "random_run"
    """Inject fault on a random repetition."""

    EVERY_RUN = "every_run"
    """Inject fault on every repetition."""

    SEQUENCE = "sequence"
    """Inject faults according to a predefined sequence of run indices."""


# ======================================================================
# 3. Scheduled Fault Config
# ======================================================================


class ScheduledFaultConfig(SerializableModel):
    """Combines a fault strategy with schedule, severity, and probability.

    This is the configuration unit for designing fault injection
    experiments.  Each instance specifies:

    - Which fault to inject (by ``strategy_name``).
    - How severe the fault should be (``severity``).
    - When to inject it (``schedule`` with optional ``run_indices``).
    - The probability of injection per opportunity (``probability``).
    - Whether the fault should be injected on the same task multiple
      times (``allow_repeated``).
    """

    strategy_name: str = Field(min_length=1)
    severity: FaultSeverity = FaultSeverity.MODERATE
    schedule: FaultSchedule = FaultSchedule.EVERY_RUN
    run_indices: list[int] = Field(default_factory=list)
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    allow_repeated: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


# ======================================================================
# 4. Fault Injection Experiment Config
# ======================================================================


class FaultInjectionConfig(SerializableModel):
    """Complete configuration for a fault injection experiment.

    This model captures all knobs a researcher can tune when designing
    a fault injection study.

    Parameters
    ----------
    enabled_faults:
        List of scheduled fault configurations.
    global_probability:
        Base probability applied to all faults (can be overridden per
        ScheduledFaultConfig).
    global_severity:
        Base severity applied to all faults (can be overridden).
    global_schedule:
        Base schedule applied to all faults (can be overridden).
    max_retries:
        Max retries per fault-injected run.
    seed:
        Random seed for reproducibility.
    repeat_faults_across_tasks:
        Whether to apply the same fault pattern across all tasks or
        randomise per task.
    """

    enabled_faults: list[ScheduledFaultConfig] = Field(default_factory=list)
    global_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    global_severity: FaultSeverity = FaultSeverity.MODERATE
    global_schedule: FaultSchedule = FaultSchedule.EVERY_RUN
    max_retries: int = Field(default=3, ge=0)
    seed: int | None = None
    repeat_faults_across_tasks: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


# ======================================================================
# 5. Recovery Metrics
# ======================================================================


class RecoveryMetrics(SerializableModel):
    """Detailed analysis of agent recovery behaviour under faults.

    Computed from a collection of FaultTraces to quantify how well an
    agent recovers from different fault types and severities.

    Parameters
    ----------
    fault_name:
        The fault strategy this metrics object describes.
    total_attempts:
        Number of fault-injected execution attempts.
    successful_recoveries:
        Number of attempts where the agent fully recovered (score = 1.0).
    partial_recoveries:
        Number of attempts where the agent partially recovered (0 < score < 1).
    failed_recoveries:
        Number of attempts where the agent did not recover (score = 0).
    recovery_rate:
        Fraction of attempts with any recovery (successful + partial) / total.
    full_recovery_rate:
        Fraction of attempts with full recovery.
    mean_recovery_latency:
        Average time (seconds) to reach any recovery outcome.
    median_recovery_latency:
        Median time (seconds) to reach any recovery outcome.
    degradation_trajectory:
        Sequence of scores ordered by attempt, showing how performance
        degrades across repeated fault exposures.
    recovery_trend:
        ``"improving"``, ``"degrading"``, or ``"stable"`` based on the
        trend of recovery rates over time.
    """

    fault_name: str = Field(min_length=1)
    total_attempts: int = Field(ge=0)
    successful_recoveries: int = Field(ge=0)
    partial_recoveries: int = Field(ge=0)
    failed_recoveries: int = Field(ge=0)
    recovery_rate: float = Field(ge=0.0, le=1.0)
    full_recovery_rate: float = Field(ge=0.0, le=1.0)
    mean_recovery_latency: float = Field(ge=0.0)
    median_recovery_latency: float = Field(ge=0.0)
    degradation_trajectory: list[float] = Field(default_factory=list)
    recovery_trend: str = Field(default="stable")

    @classmethod
    def from_traces(cls, traces: list[FaultTrace]) -> RecoveryMetrics:
        """Compute RecoveryMetrics from a list of FaultTraces.

        Parameters
        ----------
        traces:
            Non-empty list of FaultTraces for a single fault strategy.

        Returns
        -------
        RecoveryMetrics
            Aggregated recovery analysis.
        """
        if not traces:
            return cls(
                fault_name="unknown",
                total_attempts=0,
                successful_recoveries=0,
                partial_recoveries=0,
                failed_recoveries=0,
                recovery_rate=0.0,
                full_recovery_rate=0.0,
                mean_recovery_latency=0.0,
                median_recovery_latency=0.0,
            )

        fault_name = traces[0].fault_name
        n = len(traces)

        successful = sum(1 for t in traces if t.recovery_status == "success")
        partial = sum(1 for t in traces if t.recovery_status == "partial")
        failed = sum(1 for t in traces if t.recovery_status == "failed")

        recovery_rate = (successful + partial) / n if n > 0 else 0.0
        full_recovery_rate = successful / n if n > 0 else 0.0

        latencies = [t.latency_seconds for t in traces]
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
        sorted_latencies = sorted(latencies)
        median_latency = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0.0

        # Degradation trajectory: extract scores from trace details
        trajectory: list[float] = []
        for t in traces:
            score = t.details.get("score", 0.0) if t.details else 0.0
            if isinstance(score, int | float):
                trajectory.append(float(score))

        # Recovery trend: compare first vs last half
        if n >= 4:
            mid = n // 2
            first_half_rec = sum(
                1 for t in traces[:mid] if t.recovery_status in ("success", "partial")
            )
            second_half_rec = sum(
                1 for t in traces[mid:] if t.recovery_status in ("success", "partial")
            )
            rate_first = first_half_rec / mid if mid > 0 else 0.0
            rate_second = second_half_rec / (n - mid) if (n - mid) > 0 else 0.0
            if rate_second > rate_first + 0.1:
                trend = "improving"
            elif rate_first > rate_second + 0.1:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return cls(
            fault_name=fault_name,
            total_attempts=n,
            successful_recoveries=successful,
            partial_recoveries=partial,
            failed_recoveries=failed,
            recovery_rate=recovery_rate,
            full_recovery_rate=full_recovery_rate,
            mean_recovery_latency=mean_latency,
            median_recovery_latency=median_latency,
            degradation_trajectory=trajectory,
            recovery_trend=trend,
        )


class FaultSummary(SerializableModel):
    """Aggregate fault injection summary for an entire experiment.

    Combines RecoveryMetrics across all fault strategies along with
    global statistics.
    """

    total_fault_attempts: int = Field(ge=0)
    overall_recovery_rate: float = Field(ge=0.0, le=1.0)
    overall_full_recovery_rate: float = Field(ge=0.0, le=1.0)
    per_fault_metrics: dict[str, RecoveryMetrics] = Field(default_factory=dict)
    mean_recovery_latency: float = Field(ge=0.0)
    median_recovery_latency: float = Field(ge=0.0)
    configuration: FaultInjectionConfig | None = None

    @classmethod
    def from_trace_groups(
        cls,
        traces_by_fault: dict[str, list[FaultTrace]],
        config: FaultInjectionConfig | None = None,
    ) -> FaultSummary:
        """Build a FaultSummary from per-fault grouped traces.

        Parameters
        ----------
        traces_by_fault:
            Dictionary mapping fault_name → list of FaultTraces.
        config:
            Optional FaultInjectionConfig used in the experiment.

        Returns
        -------
        FaultSummary
        """
        per_fault: dict[str, RecoveryMetrics] = {}
        all_traces: list[FaultTrace] = []
        for fname, ftraces in traces_by_fault.items():
            metrics = RecoveryMetrics.from_traces(ftraces)
            per_fault[fname] = metrics
            all_traces.extend(ftraces)

        n = len(all_traces)
        if n == 0:
            return cls(
                total_fault_attempts=0,
                overall_recovery_rate=0.0,
                overall_full_recovery_rate=0.0,
                mean_recovery_latency=0.0,
                median_recovery_latency=0.0,
                configuration=config,
            )

        successful = sum(1 for t in all_traces if t.recovery_status == "success")
        partial = sum(1 for t in all_traces if t.recovery_status == "partial")
        recovery_rate = (successful + partial) / n
        full_recovery_rate = successful / n

        latencies = [t.latency_seconds for t in all_traces]
        mean_lat = sum(latencies) / len(latencies)
        sorted_lat = sorted(latencies)
        median_lat = sorted_lat[len(sorted_lat) // 2]

        return cls(
            total_fault_attempts=n,
            overall_recovery_rate=recovery_rate,
            overall_full_recovery_rate=full_recovery_rate,
            per_fault_metrics=per_fault,
            mean_recovery_latency=mean_lat,
            median_recovery_latency=median_lat,
            configuration=config,
        )
