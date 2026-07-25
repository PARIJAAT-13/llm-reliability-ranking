"""
Production-grade fault injection: OOM, crash, combined faults, MTBF/MTTR.

Extends the existing fault framework with realistic system-level fault
scenarios and reliability engineering metrics.

New strategies
--------------
1. **GpuOomFaultStrategy**       — simulate CUDA out-of-memory error
2. **RuntimeCrashFaultStrategy** — simulate process/container crash
3. **CombinedFaultStrategy**     — apply multiple sub-strategies together

New metrics
-----------
- ``compute_mtbf`` — Mean Time Between Failures from a list of FaultTraces
- ``compute_mttr`` — Mean Time To Recovery from a list of FaultTraces
"""

from __future__ import annotations

from typing import Any

from llm_reliability.reliability.faults.base import FaultInjectionStrategy, FaultTrace
from llm_reliability.reliability.faults.extensions import RecoveryMetrics

# ======================================================================
# 1. GPU Out-of-Memory Strategy
# ======================================================================


class GpuOomFaultStrategy(FaultInjectionStrategy):
    """Simulate CUDA out-of-memory (OOM) during inference.

    Mimics a ``torch.cuda.OutOfMemoryError`` by raising a RuntimeError
    with a realistic OOM message.  Optional ``memory_fraction`` parameter
    controls how much of the simulated memory budget is consumed before
    the OOM trigger, enabling light/moderate/severe gradations.
    """

    def __init__(self, memory_fraction: float = 1.0) -> None:
        self.memory_fraction = min(max(memory_fraction, 0.0), 1.0)

    @property
    def fault_name(self) -> str:
        return "gpu_oom"

    @property
    def injection_point(self) -> str:
        return "api_call"

    @property
    def description(self) -> str:
        return f"Simulate CUDA OOM at {self.memory_fraction:.0%} memory utilisation."

    def inject(self, target: Any, seed: int | None = None, **kwargs: Any) -> Any:
        if self.memory_fraction >= 0.5:
            raise RuntimeError(
                "CUDA out of memory. Tried to allocate 4.00 GiB "
                "(GPU 0 has 3.98 GiB of total capacity). "
                "See https://pytorch.org/docs/stable/notes/cuda.html"
            )
        # Light OOM: allow partial forward pass but record pressure
        return target

    def cleanup(self) -> None:
        pass


# ======================================================================
# 2. Runtime Crash Strategy
# ======================================================================


class RuntimeCrashFaultStrategy(FaultInjectionStrategy):
    """Simulate an unrecoverable process/container crash.

    Unlike transient faults, a runtime crash represents a non-retriable
    failure that requires external intervention (restart, redeploy).
    The ``crash_type`` parameter selects the simulated failure mode.
    """

    CRASH_MESSAGES: dict[str, str] = {
        "segfault": "Segmentation fault (core dumped) in inference worker.",
        "container_oom": "Container killed by OOM killer (exit code 137).",
        "process_exit": "Process exited with code -1 (SIGTERM).",
        "runtime_panic": "Runtime panic: unrecoverable error in model runner.",
    }

    def __init__(self, crash_type: str = "process_exit") -> None:
        self.crash_type = crash_type

    @property
    def fault_name(self) -> str:
        return "runtime_crash"

    @property
    def injection_point(self) -> str:
        return "agent_run"

    @property
    def description(self) -> str:
        msg = self.CRASH_MESSAGES.get(self.crash_type, "Unknown crash type.")
        return f"Simulate runtime crash ({self.crash_type}): {msg}"

    def inject(self, target: Any, seed: int | None = None, **kwargs: Any) -> Any:
        msg = self.CRASH_MESSAGES.get(
            self.crash_type,
            f"Runtime crash: unknown type '{self.crash_type}'.",
        )
        raise RuntimeError(msg)

    def cleanup(self) -> None:
        pass


# ======================================================================
# 3. Combined Fault Strategy
# ======================================================================


class CombinedFaultStrategy(FaultInjectionStrategy):
    """Apply multiple sub-strategies in sequence on the same execution.

    Useful for simulating compound failure scenarios (e.g., a timeout
    followed by a crash, or an API failure coupled with context truncation).
    The combined fault name is a ``+``-joined string of sub-strategy names.
    """

    def __init__(self, strategies: list[FaultInjectionStrategy]) -> None:
        if not strategies:
            raise ValueError("CombinedFaultStrategy requires at least one sub-strategy.")
        self.sub_strategies = strategies

    @property
    def fault_name(self) -> str:
        return "+".join(s.fault_name for s in self.sub_strategies)

    @property
    def injection_point(self) -> str:
        # Use the injection point of the first sub-strategy as primary
        return self.sub_strategies[0].injection_point

    @property
    def description(self) -> str:
        descs = "; ".join(s.description for s in self.sub_strategies)
        return f"Combined fault: {descs}"

    def inject(self, target: Any, seed: int | None = None, **kwargs: Any) -> Any:
        for i, strategy in enumerate(self.sub_strategies):
            sub_seed = (seed if seed is not None else 42) + i * 100 if seed is not None else None
            target = strategy.inject(target, seed=sub_seed, **kwargs)
        return target

    def cleanup(self) -> None:
        for strategy in self.sub_strategies:
            try:
                strategy.cleanup()
            except Exception:
                pass


# ======================================================================
# 4. MTBF / MTTR Computation
# ======================================================================


def compute_mtbf(traces: list[FaultTrace]) -> dict[str, float]:
    """Compute Mean Time Between Failures from fault traces.

    MTBF = total observation time / number of failures.

    Parameters
    ----------
    traces:
        Chronologically ordered FaultTrace list.

    Returns
    -------
    dict with keys:
        - ``mtbf_seconds``     — mean time between failures (seconds).
        - ``num_failures``     — count of failure events.
        - ``total_time_seconds`` — total observation span.
        - ``failure_rate``     — failures per second.
    """
    if len(traces) < 2:
        return {
            "mtbf_seconds": float("inf"),
            "num_failures": len(traces),
            "total_time_seconds": sum(t.latency_seconds for t in traces),
            "failure_rate": 0.0,
        }

    times = [t.latency_seconds for t in traces]
    total_time = sum(times)

    failures = sum(1 for t in traces if t.recovery_status == "failed")
    if failures == 0:
        return {
            "mtbf_seconds": float("inf"),
            "num_failures": 0,
            "total_time_seconds": total_time,
            "failure_rate": 0.0,
        }

    mtbf = total_time / failures
    return {
        "mtbf_seconds": mtbf,
        "num_failures": failures,
        "total_time_seconds": total_time,
        "failure_rate": failures / total_time if total_time > 0 else 0.0,
    }


def compute_mttr(traces: list[FaultTrace]) -> dict[str, float]:
    """Compute Mean Time To Recovery from fault traces.

    MTTR = sum of recovery latencies / number of recovery attempts.

    Parameters
    ----------
    traces:
        List of FaultTrace instances.

    Returns
    -------
    dict with keys:
        - ``mttr_seconds``     — mean time to recovery (seconds).
        - ``num_recoveries``   — count of recovery events.
        - ``total_recovery_time_seconds`` — sum of all recovery latencies.
        - ``recovery_rate``    — fraction of attempts that succeeded.
    """
    recoveries = [t for t in traces if t.recovery_status in ("success", "partial")]

    if not recoveries:
        return {
            "mttr_seconds": float("inf"),
            "num_recoveries": 0,
            "total_recovery_time_seconds": 0.0,
            "recovery_rate": 0.0,
        }

    total_time = sum(t.latency_seconds for t in recoveries)
    mttr = total_time / len(recoveries)

    return {
        "mttr_seconds": mttr,
        "num_recoveries": len(recoveries),
        "total_recovery_time_seconds": total_time,
        "recovery_rate": len(recoveries) / len(traces) if traces else 0.0,
    }


def compute_reliability_report(
    traces: list[FaultTrace],
    metrics: RecoveryMetrics | None = None,
) -> dict[str, Any]:
    """Combine MTBF, MTTR, and RecoveryMetrics into a unified reliability report.

    Parameters
    ----------
    traces:
        Fault trace list.
    metrics:
        Pre-computed RecoveryMetrics (optional).  If None, computed fresh.

    Returns
    -------
    dict combining MTBF, MTTR, and recovery metrics.
    """
    mtbf = compute_mtbf(traces)
    mttr = compute_mttr(traces)
    availability = 0.0
    if mtbf["mtbf_seconds"] != float("inf") and mttr["mttr_seconds"] != float("inf"):
        total = mtbf["mtbf_seconds"] + mttr["mttr_seconds"]
        availability = mtbf["mtbf_seconds"] / total if total > 0 else 0.0

    rec_metrics = metrics
    if rec_metrics is None and traces:
        rec_metrics = RecoveryMetrics.from_traces(traces)

    return {
        "mtbf": mtbf,
        "mttr": mttr,
        "availability": float(f"{availability:.6f}"),
        "num_traces": len(traces),
        "recovery_metrics": rec_metrics.model_dump() if rec_metrics is not None else {},
    }
