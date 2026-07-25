from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from llm_reliability.hardware.analysis import HardwareAnalysis
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord
from llm_reliability.utils.hardware_profile import HardwareProfile

logger = logging.getLogger(__name__)


def generate_hardware_summary(
    profile: HardwareProfile,
    metrics: list[MetricRecord] | None = None,
    executions: list[ExecutionRecord] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "profile_id": profile.profile_id,
        "profile_name": profile.profile_name or profile.profile_id,
        "os": f"{profile.os_name} {profile.os_version}",
        "cpu_architecture": profile.cpu_architecture,
        "cpu_cores_logical": profile.cpu_cores_logical,
        "cpu_cores_physical": profile.cpu_cores_physical,
        "ram_total_gb": profile.ram_total_gb,
        "ram_available_gb": profile.ram_available_gb,
        "gpu": profile.gpu_name or "None",
        "gpu_count": profile.gpu_count,
        "vram_total_gb": profile.vram_total_gb,
        "python_version": profile.python_version,
        "ollama_version": profile.ollama_version,
        "node_type": profile.node_type,
        "metadata": profile.metadata,
    }
    if metrics:
        summary["total_metrics"] = len(metrics)
        if metrics:
            summary["avg_reliability"] = round(
                sum(m.composite_reliability for m in metrics) / len(metrics), 4
            )
            summary["avg_success_rate"] = round(
                sum(m.success_rate for m in metrics) / len(metrics), 4
            )
    if executions:
        summary["total_executions"] = len(executions)
        success_count = sum(1 for e in executions if e.status == "success")
        summary["successful_executions"] = success_count
        summary["failed_executions"] = len(executions) - success_count
        summary["overall_success_rate"] = (
            round(success_count / len(executions), 4) if executions else 0.0
        )
    return summary


def generate_hardware_statistics(
    metrics: list[MetricRecord],
    executions: list[ExecutionRecord],
) -> dict[str, Any]:
    analysis = HardwareAnalysis()
    return {
        "reliability_by_ram": analysis.reliability_by_ram(metrics, executions),
        "reliability_by_vram": analysis.reliability_by_vram(metrics, executions),
        "latency_by_ram": analysis.latency_by_ram(executions),
        "failure_rate_by_memory": analysis.failure_rate_by_memory(executions),
        "success_rate_by_hardware": analysis.success_rate_by_hardware(metrics, executions),
        "memory_by_model": analysis.memory_by_model(executions),
        "model_ranking_by_hardware": analysis.model_ranking_by_hardware(metrics, executions),
    }


def generate_hardware_report(
    profile: HardwareProfile,
    metrics: list[MetricRecord] | None = None,
    executions: list[ExecutionRecord] | None = None,
) -> str:
    lines = [
        f"# Hardware Report: {profile.profile_name or profile.profile_id}",
        "",
        "## System Overview",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Profile ID | `{profile.profile_id}` |",
        f"| OS | {profile.os_name} {profile.os_version} |",
        f"| Architecture | {profile.cpu_architecture} |",
        f"| Logical CPUs | {profile.cpu_cores_logical} |",
        f"| Physical Cores | {profile.cpu_cores_physical or 'N/A'} |",
        f"| Total RAM | {profile.ram_total_gb} GB |",
        f"| Available RAM | {profile.ram_available_gb or 'N/A'} GB |",
        f"| GPU | {profile.gpu_name or 'None'} |",
        f"| VRAM | {profile.vram_total_gb} GB |",
        f"| Python | {profile.python_version or 'N/A'} |",
        f"| Ollama | {profile.ollama_version or 'N/A'} |",
        f"| Node Type | {profile.node_type} |",
        "",
    ]
    if metrics:
        scores = [m.composite_reliability for m in metrics]
        success_rates = [m.success_rate for m in metrics]
        lines.append("## Experiment Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Models Evaluated | {len(metrics)} |")
        lines.append(
            f"| Mean Reliability | {round(sum(scores) / len(scores), 4) if scores else 0.0} |"
        )
        lines.append(
            f"| Mean Success Rate | {round(sum(success_rates) / len(success_rates), 4) if success_rates else 0.0} |"
        )
        lines.append(f"| Reliability Range | {min(scores):.4f} – {max(scores):.4f} |")
        lines.append("")
    if executions:
        total = len(executions)
        success = sum(1 for e in executions if e.status == "success")
        lines.append("## Execution Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Executions | {total} |")
        lines.append(f"| Successful | {success} |")
        lines.append(f"| Failed | {total - success} |")
        lines.append(f"| Success Rate | {round(success / total, 4) if total else 0.0} |")
        lines.append("")
    lines.append("---")
    lines.append("*Generated by LLM Reliability Ranking Framework*")
    return "\n".join(lines)


def save_hardware_artifacts(
    profile: HardwareProfile,
    metrics: list[MetricRecord] | None = None,
    executions: list[ExecutionRecord] | None = None,
    output_dir: str | Path = "results/hardware",
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    summary = generate_hardware_summary(profile, metrics, executions)
    summary_path = out / "hardware_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["summary"] = summary_path
    if metrics and executions:
        stats = generate_hardware_statistics(metrics, executions)
        stats_path = out / "hardware_statistics.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        paths["statistics"] = stats_path
    report = generate_hardware_report(profile, metrics, executions)
    report_path = out / "hardware_report.md"
    report_path.write_text(report, encoding="utf-8")
    paths["report_md"] = report_path
    html = _markdown_to_html(report)
    html_path = out / "hardware_report.html"
    html_path.write_text(html, encoding="utf-8")
    paths["report_html"] = html_path
    return paths


def _markdown_to_html(markdown: str) -> str:
    lines = markdown.split("\n")
    html_lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Hardware Report</title>",
        "<style>body{font-family:sans-serif;max-width:960px;margin:2em auto;padding:0 1em;line-height:1.6}",
        "table{border-collapse:collapse;width:100%;margin:1em 0}",
        "th,td{border:1px solid #ddd;padding:8px;text-align:left}",
        "th{background:#f5f5f5}code{background:#f0f0f0;padding:2px 4px;border-radius:3px}</style></head><body>",
    ]
    in_table = False
    for line in lines:
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("|"):
            if not in_table:
                html_lines.append("<table>")
                in_table = True
            cells = [c.strip() for c in line.split("|")[1:-1]]
            is_header = "---" in line
            tag = "th" if is_header else "td"
            html_lines.append(f"<tr>{''.join(f'<{tag}>{c}</{tag}>' for c in cells)}</tr>")
        else:
            if in_table:
                html_lines.append("</table>")
                in_table = False
            if line.strip():
                html_lines.append(f"<p>{line}</p>")
    if in_table:
        html_lines.append("</table>")
    html_lines.append("</body></html>")
    return "\n".join(html_lines)
