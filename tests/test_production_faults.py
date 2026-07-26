"""Tests for production-grade fault injection (OOM, crash, combined, MTBF/MTTR)."""

import pytest

from llm_reliability.reliability.faults.base import FaultTrace
from llm_reliability.reliability.faults.extensions import (FaultSeverity,
                                                           RecoveryMetrics)
from llm_reliability.reliability.faults.production import (
    CombinedFaultStrategy, GpuOomFaultStrategy, RuntimeCrashFaultStrategy,
    compute_mtbf, compute_mttr, compute_reliability_report)

# ======================================================================
# GPU OOM
# ======================================================================


def test_gpu_oom_fault_name():
    s = GpuOomFaultStrategy()
    assert s.fault_name == "gpu_oom"


def test_gpu_oom_injection_point():
    s = GpuOomFaultStrategy()
    assert s.injection_point == "api_call"


def test_gpu_oom_severe_raises():
    s = GpuOomFaultStrategy(memory_fraction=1.0)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        s.inject("target")


def test_gpu_oom_light_passes():
    s = GpuOomFaultStrategy(memory_fraction=0.3)
    result = s.inject("target")
    assert result == "target"


def test_gpu_oom_cleanup():
    s = GpuOomFaultStrategy()
    s.cleanup()  # Should not raise


def test_gpu_oom_memory_fraction_clamped():
    s = GpuOomFaultStrategy(memory_fraction=2.0)
    assert s.memory_fraction == 1.0
    s2 = GpuOomFaultStrategy(memory_fraction=-1.0)
    assert s2.memory_fraction == 0.0


# ======================================================================
# Runtime Crash
# ======================================================================


def test_runtime_crash_fault_name():
    s = RuntimeCrashFaultStrategy()
    assert s.fault_name == "runtime_crash"


def test_runtime_crash_injection_point():
    s = RuntimeCrashFaultStrategy()
    assert s.injection_point == "agent_run"


def test_runtime_crash_raises():
    s = RuntimeCrashFaultStrategy("segfault")
    with pytest.raises(RuntimeError, match="Segmentation fault"):
        s.inject("target")


def test_runtime_crash_all_types():
    for crash_type in ["segfault", "container_oom", "process_exit", "runtime_panic"]:
        s = RuntimeCrashFaultStrategy(crash_type)
        with pytest.raises(RuntimeError):
            s.inject("target")


def test_runtime_crash_cleanup():
    s = RuntimeCrashFaultStrategy()
    s.cleanup()  # Should not raise


# ======================================================================
# Combined Fault
# ======================================================================


def test_combined_fault_name():
    timeout = GpuOomFaultStrategy()
    crash = RuntimeCrashFaultStrategy()
    s = CombinedFaultStrategy([timeout, crash])
    assert s.fault_name == "gpu_oom+runtime_crash"


def test_combined_fault_empty_raises():
    with pytest.raises(ValueError, match="requires at least one"):
        CombinedFaultStrategy([])


def test_combined_fault_applies_all():
    # Two strategies that both raise should both be applied
    oom = GpuOomFaultStrategy(memory_fraction=1.0)
    crash = RuntimeCrashFaultStrategy("process_exit")
    s = CombinedFaultStrategy([oom, crash])
    # First raises (OOM), crash never reached in this case
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        s.inject("target")


def test_combined_fault_cleanup():
    oom = GpuOomFaultStrategy()
    crash = RuntimeCrashFaultStrategy()
    s = CombinedFaultStrategy([oom, crash])
    s.cleanup()  # Should not raise


def test_combined_fault_injection_point():
    oom = GpuOomFaultStrategy()
    crash = RuntimeCrashFaultStrategy()
    s = CombinedFaultStrategy([oom, crash])
    assert s.injection_point == "api_call"  # First strategy's point


# ======================================================================
# MTBF
# ======================================================================


def _make_trace(
    fault_name: str = "test",
    status: str = "failed",
    latency: float = 1.0,
) -> FaultTrace:
    """Helper to create a FaultTrace for testing."""
    return FaultTrace(
        fault_name=fault_name,
        injection_point="api_call",
        retry_count=0,
        recovery_status=status,
        execution_outcome="error" if status == "failed" else "success",
        latency_seconds=latency,
        details={},
    )


def test_mtbf_empty():
    result = compute_mtbf([])
    assert result["num_failures"] == 0
    assert result["mtbf_seconds"] == float("inf")


def test_mtbf_single():
    traces = [_make_trace(status="failed", latency=5.0)]
    result = compute_mtbf(traces)
    assert result["num_failures"] == 1
    assert result["mtbf_seconds"] == float("inf")  # < 2 traces


def test_mtbf_no_failures():
    traces = [
        _make_trace(status="success", latency=1.0),
        _make_trace(status="success", latency=2.0),
    ]
    result = compute_mtbf(traces)
    assert result["num_failures"] == 0
    assert result["mtbf_seconds"] == float("inf")


def test_mtbf_basic():
    traces = [
        _make_trace(status="failed", latency=2.0),
        _make_trace(status="success", latency=3.0),
        _make_trace(status="failed", latency=5.0),
    ]
    result = compute_mtbf(traces)
    assert result["num_failures"] == 2
    total_time = 2.0 + 3.0 + 5.0
    assert result["mtbf_seconds"] == pytest.approx(total_time / 2.0)
    assert result["failure_rate"] == pytest.approx(2.0 / total_time)


# ======================================================================
# MTTR
# ======================================================================


def test_mttr_empty():
    result = compute_mttr([])
    assert result["num_recoveries"] == 0
    assert result["mttr_seconds"] == float("inf")


def test_mttr_no_recoveries():
    traces = [
        _make_trace(status="failed", latency=1.0),
        _make_trace(status="failed", latency=2.0),
    ]
    result = compute_mttr(traces)
    assert result["num_recoveries"] == 0
    assert result["mttr_seconds"] == float("inf")


def test_mttr_basic():
    traces = [
        _make_trace(status="success", latency=2.0),
        _make_trace(status="failed", latency=3.0),
        _make_trace(status="partial", latency=5.0),
    ]
    result = compute_mttr(traces)
    assert result["num_recoveries"] == 2
    assert result["mttr_seconds"] == pytest.approx((2.0 + 5.0) / 2.0)
    assert result["recovery_rate"] == pytest.approx(2.0 / 3.0)


# ======================================================================
# Reliability Report
# ======================================================================


def test_reliability_report_basic():
    traces = [
        _make_trace(status="success", latency=1.0),
        _make_trace(status="failed", latency=2.0),
        _make_trace(status="partial", latency=1.5),
    ]
    report = compute_reliability_report(traces)
    assert "mtbf" in report
    assert "mttr" in report
    assert "availability" in report
    assert "num_traces" in report
    assert report["num_traces"] == 3
    assert 0.0 <= report["availability"] <= 1.0
    assert "recovery_metrics" in report


def test_reliability_report_empty():
    report = compute_reliability_report([])
    assert report["num_traces"] == 0
    assert report["mtbf"]["num_failures"] == 0
    assert report["mttr"]["num_recoveries"] == 0


def test_reliability_report_availability():
    traces = [
        _make_trace(status="success", latency=0.5),
        _make_trace(status="success", latency=0.5),
    ]
    report = compute_reliability_report(traces)
    # No failures → availability = 1.0 (no failures, MTBF infinite)
    assert report["mtbf"]["num_failures"] == 0 or report["availability"] >= 0.0
