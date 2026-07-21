"""
Purpose
-------
Generate fault tolerance summaries and reporting artifacts.

Responsibilities
----------------
- Aggregate FaultTrace metrics across tasks and experiments.
- Compute fault type frequency, recovery rate, failure rate, and average recovery latency.
- Format fault reports into canonical JSON and markdown table artifacts.

Design notes
------------
FaultReport collects agent resilience metrics across fault strategies.
It enables researchers to audit agent vulnerability profiles across specific failure modes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from llm_reliability.reliability.faults.base import FaultRunResult, FaultTrace
from llm_reliability.utils.serialization import SerializableModel


class FaultTypeMetrics(SerializableModel):
    """Aggregated fault tolerance metrics for a specific fault strategy."""

    fault_name: str = Field(min_length=1)
    injection_point: str = Field(min_length=1)
    attempts: int = Field(ge=0)
    recovered_count: int = Field(ge=0)
    successful_recovery_count: int = Field(ge=0)
    partial_recovery_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    recovery_rate: float = Field(ge=0.0, le=1.0)
    failure_rate: float = Field(ge=0.0, le=1.0)
    avg_recovery_latency_seconds: float = Field(ge=0.0)


class FaultReport(SerializableModel):
    """Canonical summary report for fault tolerance experiment runs."""

    total_fault_attempts: int = Field(ge=0)
    overall_recovery_rate: float = Field(ge=0.0, le=1.0)
    overall_failure_rate: float = Field(ge=0.0, le=1.0)
    by_fault_type: dict[str, FaultTypeMetrics] = Field(default_factory=dict)
    fault_traces: list[FaultTrace] = Field(default_factory=list)
    generated_at: str = Field(min_length=1)

    def to_markdown(self) -> str:
        """Format fault report as a Markdown summary table."""
        lines = [
            "# Fault Tolerance Summary Report",
            "",
            f"- **Generated At**: {self.generated_at}",
            f"- **Total Fault Attempts**: {self.total_fault_attempts}",
            f"- **Overall Recovery Rate**: {self.overall_recovery_rate:.2%}",
            f"- **Overall Failure Rate**: {self.overall_failure_rate:.2%}",
            "",
            "## Breakdown by Fault Type",
            "",
            "| Fault Type | Injection Point | Attempts | Recovered | Failed | Recovery Rate | Failure Rate | Avg Latency (s) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for fname, metrics in sorted(self.by_fault_type.items()):
            lines.append(
                f"| {metrics.fault_name} | {metrics.injection_point} | {metrics.attempts} | "
                f"{metrics.recovered_count} | {metrics.failure_count} | "
                f"{metrics.recovery_rate:.2%} | {metrics.failure_rate:.2%} | {metrics.avg_recovery_latency_seconds:.3f} |"
            )

        return "\n".join(lines)


class FaultReportGenerator:
    """Aggregates fault traces into structured FaultReport objects."""

    @staticmethod
    def generate_report(
        results: FaultRunResult | list[FaultRunResult] | list[FaultTrace],
    ) -> FaultReport:
        """Generate a FaultReport from FaultRunResult(s) or FaultTrace list.

        Parameters
        ----------
        results : FaultRunResult | list[FaultRunResult] | list[FaultTrace]
            Experiment results or trace data.

        Returns
        -------
        FaultReport
            Aggregated summary report.
        """
        traces: list[FaultTrace] = []
        if isinstance(results, FaultRunResult):
            traces = results.fault_traces
        elif isinstance(results, list):
            for item in results:
                if isinstance(item, FaultRunResult):
                    traces.extend(item.fault_traces)
                elif isinstance(item, FaultTrace):
                    traces.append(item)

        if not traces:
            return FaultReport(
                total_fault_attempts=0,
                overall_recovery_rate=0.0,
                overall_failure_rate=0.0,
                by_fault_type={},
                fault_traces=[],
                generated_at=datetime.now(timezone.utc).isoformat(),
            )

        grouped: dict[str, list[FaultTrace]] = {}
        for trace in traces:
            grouped.setdefault(trace.fault_name, []).append(trace)

        by_fault_type: dict[str, FaultTypeMetrics] = {}
        total_recovered = 0
        total_failed = 0

        for fname, group in grouped.items():
            attempts = len(group)
            succ = sum(1 for t in group if t.recovery_status == "success")
            part = sum(1 for t in group if t.recovery_status == "partial")
            fail = sum(1 for t in group if t.recovery_status == "failed")
            rec = succ + part

            total_recovered += rec
            total_failed += fail

            rec_rate = rec / attempts if attempts > 0 else 0.0
            fail_rate = fail / attempts if attempts > 0 else 0.0

            rec_latencies = [t.latency_seconds for t in group if t.recovery_status in ("success", "partial")]
            avg_lat = sum(rec_latencies) / len(rec_latencies) if rec_latencies else 0.0

            by_fault_type[fname] = FaultTypeMetrics(
                fault_name=fname,
                injection_point=group[0].injection_point,
                attempts=attempts,
                recovered_count=rec,
                successful_recovery_count=succ,
                partial_recovery_count=part,
                failure_count=fail,
                recovery_rate=rec_rate,
                failure_rate=fail_rate,
                avg_recovery_latency_seconds=avg_lat,
            )

        total_attempts = len(traces)
        overall_rec = total_recovered / total_attempts if total_attempts > 0 else 0.0
        overall_fail = total_failed / total_attempts if total_attempts > 0 else 0.0

        return FaultReport(
            total_fault_attempts=total_attempts,
            overall_recovery_rate=overall_rec,
            overall_failure_rate=overall_fail,
            by_fault_type=by_fault_type,
            fault_traces=traces,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
