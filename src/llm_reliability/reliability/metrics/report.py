"""
Purpose
-------
Generate quantitative reliability metric reports across per-agent, per-benchmark, and overall scopes.

Responsibilities
----------------
- Aggregate metric results computed by ReliabilityMetricsEngine into a canonical ReliabilityMetricReport.
- Provide Markdown formatting methods for paper documentation and CLI reporting.
- Export MetricRecord objects compatible with the framework ranking pipeline.

Design notes
------------
ReliabilityMetricReport wraps structured result outputs from engine calculations into a single,
self-contained, serializable artifact.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from llm_reliability.records.metric import MetricRecord
from llm_reliability.utils.serialization import SerializableModel


class ReliabilityMetricReport(SerializableModel):
    """Canonical report containing per-agent, per-benchmark, and overall reliability metrics."""

    per_agent_metrics: dict[str, Any] = Field(default_factory=dict)
    per_benchmark_metrics: dict[str, Any] = Field(default_factory=dict)
    overall_metrics: dict[str, Any] = Field(default_factory=dict)
    metric_records: list[MetricRecord] = Field(default_factory=list)
    generated_at: str = Field(min_length=1)

    def to_markdown(self) -> str:
        """Format the report into a clean Markdown table representation."""
        lines = [
            "# Quantitative LLM Reliability Evaluation Report",
            "",
            f"- **Generated At**: {self.generated_at}",
            f"- **Agents Evaluated**: {len(self.per_agent_metrics)}",
            f"- **Benchmarks**: {len(self.per_benchmark_metrics)}",
            "",
            "## Per-Agent Reliability Metrics",
            "",
            "| Agent | Success Rate | Consistency | Robustness | Fault Tolerance | Composite Score |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        for agent_name, summary in sorted(self.per_agent_metrics.items()):
            c_score = summary.get("consistency", {}).get("deterministic_consistency_score", 0.0)
            r_score = summary.get("robustness", {}).get("robustness_score", 0.0)
            f_score = summary.get("fault_tolerance", {}).get("fault_tolerance_score", 0.0)
            sr = summary.get("consistency", {}).get("success_rate", 0.0)
            comp = summary.get("composite_score", 0.0)

            lines.append(
                f"| {agent_name} | {sr:.2%} | {c_score:.4f} | {r_score:.4f} | {f_score:.4f} | **{comp:.4f}** |"
            )

        lines.extend(
            [
                "",
                "## Per-Benchmark Metrics",
                "",
                "| Benchmark | Success Rate | Consistency | Robustness | Fault Tolerance | Composite Score |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )

        for bench_name, summary in sorted(self.per_benchmark_metrics.items()):
            c_score = summary.get("consistency", {}).get("deterministic_consistency_score", 0.0)
            r_score = summary.get("robustness", {}).get("robustness_score", 0.0)
            f_score = summary.get("fault_tolerance", {}).get("fault_tolerance_score", 0.0)
            sr = summary.get("consistency", {}).get("success_rate", 0.0)
            comp = summary.get("composite_score", 0.0)

            lines.append(
                f"| {bench_name} | {sr:.2%} | {c_score:.4f} | {r_score:.4f} | {f_score:.4f} | **{comp:.4f}** |"
            )

        return "\n".join(lines)


class ReliabilityReportGenerator:
    """Factory for generating structured ReliabilityMetricReport artifacts."""

    @staticmethod
    def generate_report(engine_output: dict[str, Any]) -> ReliabilityMetricReport:
        """Create a ReliabilityMetricReport from ReliabilityMetricsEngine output.

        Parameters
        ----------
        engine_output : dict[str, Any]
            Dictionary output from ReliabilityMetricsEngine.compute_all().

        Returns
        -------
        ReliabilityMetricReport
            Canonical metric report.
        """
        per_agent_raw = engine_output.get("per_agent", {})
        per_bench_raw = engine_output.get("per_benchmark", {})
        overall_raw = engine_output.get("overall", {})
        m_records = engine_output.get("metric_records", [])

        per_agent_dict: dict[str, Any] = {}
        for k, v in per_agent_raw.items():
            per_agent_dict[k] = v.to_dict() if hasattr(v, "to_dict") else v

        per_bench_dict: dict[str, Any] = {}
        for k, v in per_bench_raw.items():
            per_bench_dict[k] = v.to_dict() if hasattr(v, "to_dict") else v

        overall_dict = overall_raw.to_dict() if hasattr(overall_raw, "to_dict") else overall_raw

        return ReliabilityMetricReport(
            per_agent_metrics=per_agent_dict,
            per_benchmark_metrics=per_bench_dict,
            overall_metrics=overall_dict,
            metric_records=m_records,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
