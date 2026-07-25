"""Tests for the fault injection framework extensions."""

import pytest

from llm_reliability.reliability.faults.base import FaultTrace
from llm_reliability.reliability.faults.extensions import (
    SEVERITY_PARAMS,
    FaultInjectionConfig,
    FaultSchedule,
    FaultSeverity,
    FaultSummary,
    RecoveryMetrics,
    ScheduledFaultConfig,
)

# ======================================================================
# FaultSeverity
# ======================================================================


def test_severity_enum_values():
    assert FaultSeverity.LIGHT.value == "light"
    assert FaultSeverity.MODERATE.value == "moderate"
    assert FaultSeverity.SEVERE.value == "severe"
    assert FaultSeverity.CRITICAL.value == "critical"


def test_severity_params_exist_for_all():
    for severity in FaultSeverity:
        assert severity in SEVERITY_PARAMS
        params = SEVERITY_PARAMS[severity]
        assert "truncation_ratio" in params
        assert "delay_seconds" in params
        assert "description" in params


def test_severity_params_monotonic():
    """Truncation and delay should increase with severity."""
    ratios = [
        SEVERITY_PARAMS[s]["truncation_ratio"]
        for s in [
            FaultSeverity.LIGHT,
            FaultSeverity.MODERATE,
            FaultSeverity.SEVERE,
            FaultSeverity.CRITICAL,
        ]
    ]
    assert ratios == sorted(ratios)


def test_severity_params_bounds():
    for severity in FaultSeverity:
        params = SEVERITY_PARAMS[severity]
        assert 0.0 <= params["truncation_ratio"] <= 1.0
        assert params["delay_seconds"] >= 0.0


# ======================================================================
# FaultSchedule
# ======================================================================


def test_schedule_enum_values():
    assert FaultSchedule.FIRST_RUN.value == "first_run"
    assert FaultSchedule.LAST_RUN.value == "last_run"
    assert FaultSchedule.RANDOM_RUN.value == "random_run"
    assert FaultSchedule.EVERY_RUN.value == "every_run"
    assert FaultSchedule.SEQUENCE.value == "sequence"


# ======================================================================
# ScheduledFaultConfig
# ======================================================================


def test_scheduled_fault_config_defaults():
    cfg = ScheduledFaultConfig(strategy_name="timeout")
    assert cfg.strategy_name == "timeout"
    assert cfg.severity == FaultSeverity.MODERATE
    assert cfg.schedule == FaultSchedule.EVERY_RUN
    assert cfg.probability == 1.0
    assert cfg.allow_repeated


def test_scheduled_fault_config_custom():
    cfg = ScheduledFaultConfig(
        strategy_name="api_failure",
        severity=FaultSeverity.CRITICAL,
        schedule=FaultSchedule.FIRST_RUN,
        probability=0.5,
        allow_repeated=False,
    )
    assert cfg.strategy_name == "api_failure"
    assert cfg.severity == FaultSeverity.CRITICAL
    assert cfg.schedule == FaultSchedule.FIRST_RUN
    assert cfg.probability == 0.5
    assert not cfg.allow_repeated


def test_scheduled_fault_config_probability_bounds():
    with pytest.raises(Exception):
        ScheduledFaultConfig(strategy_name="x", probability=-0.1)
    with pytest.raises(Exception):
        ScheduledFaultConfig(strategy_name="x", probability=1.5)


def test_scheduled_fault_config_sequence_indices():
    cfg = ScheduledFaultConfig(
        strategy_name="timeout",
        schedule=FaultSchedule.SEQUENCE,
        run_indices=[0, 2, 4],
    )
    assert cfg.run_indices == [0, 2, 4]


# ======================================================================
# FaultInjectionConfig
# ======================================================================


def test_fault_injection_config_defaults():
    cfg = FaultInjectionConfig()
    assert cfg.global_probability == 1.0
    assert cfg.global_severity == FaultSeverity.MODERATE
    assert cfg.global_schedule == FaultSchedule.EVERY_RUN
    assert cfg.max_retries == 3
    assert cfg.enabled_faults == []


def test_fault_injection_config_with_faults():
    cfg = FaultInjectionConfig(
        enabled_faults=[
            ScheduledFaultConfig(strategy_name="timeout"),
            ScheduledFaultConfig(
                strategy_name="api_failure",
                severity=FaultSeverity.SEVERE,
            ),
        ],
        global_probability=0.8,
        max_retries=5,
        seed=42,
    )
    assert len(cfg.enabled_faults) == 2
    assert cfg.global_probability == 0.8
    assert cfg.max_retries == 5
    assert cfg.seed == 42


# ======================================================================
# RecoveryMetrics
# ======================================================================


def _make_trace(
    fault_name: str = "timeout",
    recovery_status: str = "success",
    latency: float = 1.0,
    score: float = 1.0,
) -> FaultTrace:
    return FaultTrace(
        fault_name=fault_name,
        injection_point="agent_run",
        retry_count=0,
        recovery_status=recovery_status,  # type: ignore[arg-type]
        execution_outcome="success",
        latency_seconds=latency,
        details={"score": score},
    )


def test_recovery_metrics_empty():
    metrics = RecoveryMetrics.from_traces([])
    assert metrics.total_attempts == 0
    assert metrics.recovery_rate == 0.0
    assert metrics.full_recovery_rate == 0.0


def test_recovery_metrics_all_success():
    traces = [_make_trace("timeout") for _ in range(5)]
    metrics = RecoveryMetrics.from_traces(traces)
    assert metrics.fault_name == "timeout"
    assert metrics.total_attempts == 5
    assert metrics.successful_recoveries == 5
    assert metrics.recovery_rate == 1.0
    assert metrics.full_recovery_rate == 1.0


def test_recovery_metrics_all_failed():
    traces = [_make_trace("timeout", recovery_status="failed", score=0.0) for _ in range(3)]
    metrics = RecoveryMetrics.from_traces(traces)
    assert metrics.total_attempts == 3
    assert metrics.failed_recoveries == 3
    assert metrics.recovery_rate == 0.0
    assert metrics.full_recovery_rate == 0.0


def test_recovery_metrics_mixed():
    traces = [
        _make_trace("timeout", recovery_status="success", score=1.0),
        _make_trace("timeout", recovery_status="partial", score=0.5),
        _make_trace("timeout", recovery_status="failed", score=0.0),
        _make_trace("timeout", recovery_status="success", score=1.0),
    ]
    metrics = RecoveryMetrics.from_traces(traces)
    assert metrics.total_attempts == 4
    assert metrics.successful_recoveries == 2
    assert metrics.partial_recoveries == 1
    assert metrics.failed_recoveries == 1
    assert metrics.recovery_rate == 0.75
    assert metrics.full_recovery_rate == 0.5


def test_recovery_metrics_latency():
    traces = [
        _make_trace("timeout", latency=1.0, score=1.0),
        _make_trace("timeout", latency=2.0, score=1.0),
        _make_trace("timeout", latency=3.0, score=1.0),
    ]
    metrics = RecoveryMetrics.from_traces(traces)
    assert metrics.mean_recovery_latency == 2.0
    assert metrics.median_recovery_latency == 2.0


def test_recovery_metrics_degradation_trajectory():
    traces = [
        _make_trace("timeout", score=1.0),
        _make_trace("timeout", score=0.7),
        _make_trace("timeout", score=0.3),
        _make_trace("timeout", score=0.0),
    ]
    metrics = RecoveryMetrics.from_traces(traces)
    assert metrics.degradation_trajectory == [1.0, 0.7, 0.3, 0.0]


def test_recovery_metrics_trend_degrading():
    traces = [
        _make_trace("timeout", recovery_status="success", score=1.0),
        _make_trace("timeout", recovery_status="success", score=1.0),
        _make_trace("timeout", recovery_status="failed", score=0.0),
        _make_trace("timeout", recovery_status="failed", score=0.0),
    ]
    metrics = RecoveryMetrics.from_traces(traces)
    assert metrics.recovery_trend == "degrading"


def test_recovery_metrics_trend_improving():
    traces = [
        _make_trace("timeout", recovery_status="failed", score=0.0),
        _make_trace("timeout", recovery_status="failed", score=0.0),
        _make_trace("timeout", recovery_status="success", score=1.0),
        _make_trace("timeout", recovery_status="success", score=1.0),
    ]
    metrics = RecoveryMetrics.from_traces(traces)
    assert metrics.recovery_trend == "improving"


def test_recovery_metrics_trend_stable():
    traces = [
        _make_trace("timeout", recovery_status="success", score=1.0),
        _make_trace("timeout", recovery_status="success", score=1.0),
        _make_trace("timeout", recovery_status="success", score=1.0),
        _make_trace("timeout", recovery_status="success", score=1.0),
    ]
    metrics = RecoveryMetrics.from_traces(traces)
    assert metrics.recovery_trend == "stable"


# ======================================================================
# FaultSummary
# ======================================================================


def test_fault_summary_empty():
    summary = FaultSummary.from_trace_groups({})
    assert summary.total_fault_attempts == 0
    assert summary.overall_recovery_rate == 0.0


def test_fault_summary_single_fault():
    traces_by_fault = {
        "timeout": [
            _make_trace("timeout", score=1.0),
            _make_trace("timeout", score=0.0),
        ]
    }
    summary = FaultSummary.from_trace_groups(traces_by_fault)
    assert summary.total_fault_attempts == 2
    assert "timeout" in summary.per_fault_metrics
    assert summary.per_fault_metrics["timeout"].total_attempts == 2


def test_fault_summary_multiple_faults():
    traces_by_fault = {
        "timeout": [_make_trace("timeout", score=1.0) for _ in range(3)],
        "api_failure": [
            _make_trace("api_failure", recovery_status="failed", score=0.0) for _ in range(2)
        ],
    }
    summary = FaultSummary.from_trace_groups(traces_by_fault)
    assert summary.total_fault_attempts == 5
    assert "timeout" in summary.per_fault_metrics
    assert "api_failure" in summary.per_fault_metrics
    assert summary.per_fault_metrics["timeout"].recovery_rate == 1.0
    assert summary.per_fault_metrics["api_failure"].recovery_rate == 0.0


def test_fault_summary_with_config():
    config = FaultInjectionConfig(seed=42)
    traces_by_fault = {
        "timeout": [_make_trace("timeout", score=1.0)],
    }
    summary = FaultSummary.from_trace_groups(traces_by_fault, config=config)
    assert summary.configuration is not None
    assert summary.configuration.seed == 42


def test_fault_summary_latencies():
    traces_by_fault = {
        "timeout": [
            _make_trace("timeout", latency=1.0, score=1.0),
            _make_trace("timeout", latency=3.0, score=1.0),
            _make_trace("timeout", latency=2.0, score=1.0),
        ]
    }
    summary = FaultSummary.from_trace_groups(traces_by_fault)
    assert summary.mean_recovery_latency == 2.0
    assert summary.median_recovery_latency == 2.0
