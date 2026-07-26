"""
Publication reporting — automated generation of publication-ready artifacts.

Generates experiment_summary.json, runtime_summary.json, hardware_summary.json,
benchmark_summary.json, ranking_summary.json, statistics_summary.json,
LaTeX tables, Markdown tables, CSV summaries, and reproducibility manifests.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_reliability.experiments.extended_models import ReproducibilityManifest
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord
from llm_reliability.utils.hardware_profile import detect_hardware_profile

logger = logging.getLogger(__name__)


def generate_experiment_summary(
    experiment_id: str,
    metrics: list[MetricRecord],
    rankings: list[RankingRecord],
    executions: list[ExecutionRecord],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a comprehensive experiment summary JSON."""
    agents = sorted(set(m.agent for m in metrics))
    benchmarks = sorted(set(m.benchmark for m in metrics))
    n_evals = len(executions)
    n_metrics = len(metrics)
    reliability_scores = [m.composite_reliability for m in metrics]
    success_rates = [m.success_rate for m in metrics]
    summary: dict[str, Any] = {
        "experiment_id": experiment_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agents": agents,
        "benchmarks": benchmarks,
        "total_executions": n_evals,
        "total_metrics": n_metrics,
        "reliability": {
            "mean": (
                round(sum(reliability_scores) / len(reliability_scores), 4)
                if reliability_scores
                else 0.0
            ),
            "min": round(min(reliability_scores), 4) if reliability_scores else 0.0,
            "max": round(max(reliability_scores), 4) if reliability_scores else 0.0,
            "std": _std(reliability_scores),
        },
        "success_rate": {
            "mean": (round(sum(success_rates) / len(success_rates), 4) if success_rates else 0.0),
            "min": round(min(success_rates), 4) if success_rates else 0.0,
            "max": round(max(success_rates), 4) if success_rates else 0.0,
            "std": _std(success_rates),
        },
        "rankings": {
            r.ranking_type: [
                {"rank": rank, "agent": agent, "score": score}
                for rank, (agent, score) in enumerate(r.rankings, 1)
            ]
            for r in rankings
        },
    }
    if config:
        summary["configuration"] = config
    return summary


def generate_runtime_summary(
    runtime_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a runtime summary from detected or provided metadata."""
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if runtime_metadata:
        summary["runtime"] = runtime_metadata
    import sys

    summary["python"] = {
        "version": sys.version,
        "executable": sys.executable,
    }
    return summary


def generate_benchmark_summary(
    metrics: list[MetricRecord],
    executions: list[ExecutionRecord],
) -> list[dict[str, Any]]:
    """Generate per-benchmark summary statistics."""
    bench_groups: dict[str, dict[str, Any]] = {}
    for m in metrics:
        b = m.benchmark
        if b not in bench_groups:
            bench_groups[b] = {
                "benchmark": b,
                "agents": set(),
                "total_executions": 0,
                "reliability_scores": [],
                "success_rates": [],
            }
        bench_groups[b]["agents"].add(m.agent)
        bench_groups[b]["reliability_scores"].append(m.composite_reliability)
        bench_groups[b]["success_rates"].append(m.success_rate)
    for e in executions:
        b = e.benchmark
        if b in bench_groups:
            bench_groups[b]["total_executions"] += 1
    results = []
    for b, data in sorted(bench_groups.items()):
        scores = data["reliability_scores"]
        rates = data["success_rates"]
        results.append(
            {
                "benchmark": b,
                "agents": sorted(data["agents"]),
                "total_executions": data["total_executions"],
                "mean_reliability": (round(sum(scores) / len(scores), 4) if scores else 0.0),
                "mean_success_rate": (round(sum(rates) / len(rates), 4) if rates else 0.0),
            }
        )
    return results


def generate_ranking_summary(rankings: list[RankingRecord]) -> list[dict[str, Any]]:
    """Generate per-ranking-type summary."""
    return [
        {
            "ranking_type": r.ranking_type,
            "benchmark": r.benchmark,
            "rankings": [
                {"rank": rank, "agent": agent, "score": score}
                for rank, (agent, score) in enumerate(r.rankings, 1)
            ],
        }
        for r in rankings
    ]


def generate_statistics_summary(metrics: list[MetricRecord]) -> dict[str, Any]:
    """Generate aggregate statistics over all metrics."""
    scores = [m.composite_reliability for m in metrics]
    rates = [m.success_rate for m in metrics]
    return {
        "model_count": len(metrics),
        "reliability": {
            "mean": _safe_mean(scores),
            "median": _safe_median(scores),
            "std": _std(scores),
            "min": min(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
            "q25": _percentile(scores, 25),
            "q75": _percentile(scores, 75),
        },
        "success_rate": {
            "mean": _safe_mean(rates),
            "median": _safe_median(rates),
            "std": _std(rates),
            "min": min(rates) if rates else 0.0,
            "max": max(rates) if rates else 0.0,
        },
    }


def generate_reproducibility_manifest(
    experiment_id: str,
    config: dict[str, Any] | None = None,
    seeds: list[int] | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> ReproducibilityManifest:
    """Generate a complete reproducibility manifest."""
    import llm_reliability

    git_hash = _get_git_commit_hash()
    hw = detect_hardware_profile(profile_id="manifest")
    env_vars = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("SECRET") and not k.startswith("API")
    }
    deps = _get_dependency_versions()

    return ReproducibilityManifest(
        experiment_id=experiment_id,
        framework_version=getattr(llm_reliability, "__version__", "0.0.0"),
        git_commit_hash=git_hash,
        python_version=platform.python_version(),
        operating_system=f"{platform.system()} {platform.release()}",
        configuration=config or {},
        hardware_profile=hw.canonical_dict() if hw else {},
        random_seeds=seeds or [],
        environment_variables=env_vars,
        artifact_checksums=artifact_paths or {},
        experiment_timestamp=datetime.now(timezone.utc).isoformat(),
        dependencies=deps,
    )


def generate_latex_table(
    rankings: list[RankingRecord],
    caption: str = "Model rankings across benchmarks.",
    label: str = "tab:rankings",
) -> str:
    """Generate a LaTeX table from ranking records."""
    if not rankings:
        return "% No ranking data available."
    benchmarks = sorted(set(r.benchmark for r in rankings))
    agents_set: set[str] = set()
    for r in rankings:
        for agent, _ in r.rankings:
            agents_set.add(agent)
    agents = sorted(agents_set)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(benchmarks) + "}",
        "\\toprule",
        "Agent & " + " & ".join(b.replace("_", "\\_") for b in benchmarks) + " \\\\",
        "\\midrule",
    ]
    for agent in agents:
        row = [agent.replace("_", "\\_")]
        for b in benchmarks:
            score = ""
            for r in rankings:
                if r.benchmark == b:
                    for a, s in r.rankings:
                        if a == agent:
                            score = f"{s:.3f}"
                            break
            row.append(score)
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_markdown_table(
    rankings: list[RankingRecord],
) -> str:
    """Generate a Markdown table from ranking records."""
    if not rankings:
        return "*No ranking data available.*"
    benchmarks = sorted(set(r.benchmark for r in rankings))
    agents_set: set[str] = set()
    for r in rankings:
        for agent, _ in r.rankings:
            agents_set.add(agent)
    agents = sorted(agents_set)
    header = "| Agent | " + " | ".join(b for b in benchmarks) + " |"
    sep = "|-------|" + "|".join("-------" for _ in benchmarks) + "|"
    lines = [header, sep]
    for agent in agents:
        row = [f" **{agent}** "]
        for b in benchmarks:
            score = ""
            for r in rankings:
                if r.benchmark == b:
                    for a, s in r.rankings:
                        if a == agent:
                            score = f"{s:.4f}"
                            break
            row.append(f" {score} " if score else " — ")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_csv(rankings: list[RankingRecord], output_path: str | Path) -> Path:
    """Write a CSV summary of rankings."""
    import csv

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ranking_type", "benchmark", "rank", "agent", "score"])
        for r in rankings:
            for rank, (agent, score) in enumerate(r.rankings, 1):
                writer.writerow([r.ranking_type, r.benchmark, rank, agent, score])
    return path


def save_publication_artifacts(
    experiment_id: str,
    metrics: list[MetricRecord],
    rankings: list[RankingRecord],
    executions: list[ExecutionRecord],
    config: dict[str, Any] | None = None,
    output_dir: str | Path = "results/publication",
) -> dict[str, Path]:
    """Generate and save all publication-ready artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    exp_summary = generate_experiment_summary(experiment_id, metrics, rankings, executions, config)
    paths["experiment_summary"] = _write_json(out / "experiment_summary.json", exp_summary)

    runtime_summary = generate_runtime_summary()
    paths["runtime_summary"] = _write_json(out / "runtime_summary.json", runtime_summary)

    hw = detect_hardware_profile(profile_id="publication")
    hw_dict = hw.canonical_dict() if hw else {}
    paths["hardware_summary"] = _write_json(out / "hardware_summary.json", hw_dict)

    bench_summary = generate_benchmark_summary(metrics, executions)
    paths["benchmark_summary"] = _write_json(out / "benchmark_summary.json", bench_summary)

    ranking_summary = generate_ranking_summary(rankings)
    paths["ranking_summary"] = _write_json(out / "ranking_summary.json", ranking_summary)

    stats_summary = generate_statistics_summary(metrics)
    paths["statistics_summary"] = _write_json(out / "statistics_summary.json", stats_summary)

    latex = generate_latex_table(rankings)
    paths["latex_table"] = _write_text(out / "rankings.tex", latex)

    md = generate_markdown_table(rankings)
    paths["markdown_table"] = _write_text(out / "rankings.md", md)

    csv_path = generate_csv(rankings, out / "rankings.csv")
    paths["csv"] = csv_path

    manifest = generate_reproducibility_manifest(
        experiment_id,
        config,
        seeds=None,
        artifact_paths={k: str(v) for k, v in paths.items()},
    )
    paths["manifest"] = _write_json(
        out / "reproducibility_manifest.json", manifest.canonical_dict()
    )

    return paths


def _write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _safe_mean(vals: list[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _safe_median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n % 2 == 0:
        return round((s[n // 2 - 1] + s[n // 2]) / 2, 4)
    return round(s[n // 2], 4)


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    import statistics

    return round(statistics.stdev(vals), 4)


def _percentile(vals: list[float], p: int) -> float:
    if not vals:
        return 0.0
    import statistics

    return round(statistics.quantiles(vals, n=100)[p - 1], 4)


def _get_git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_dependency_versions() -> dict[str, str]:
    deps: dict[str, str] = {}
    try:
        import pkg_resources

        for dist in pkg_resources.working_set:
            deps[dist.key] = dist.version
    except Exception:
        pass
    return deps
