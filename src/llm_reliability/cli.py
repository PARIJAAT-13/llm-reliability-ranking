"""
Command-line interface for the LLM Reliability Ranking framework.

Supports experiment execution, resume, checkpoint, compare, report, export,
validate, discovery, system info, hardware info, and statistics commands.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from llm_reliability import __version__
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry
from llm_reliability.runtime.registry import RuntimeRegistry

logger = logging.getLogger(__name__)


def _show_version(args: argparse.Namespace) -> None:
    print(f"llm-reliability-ranking v{__version__}")


def _run_experiment(args: argparse.Namespace) -> None:
    path = Path(args.config)
    if not path.exists():
        print(f"Error: config file not found: {path}", file=sys.stderr)
        sys.exit(1)

    from llm_reliability.agents.agent_factory import AgentFactory
    from llm_reliability.cache import ExperimentCache
    from llm_reliability.experiments.experiment_models import ExperimentSpec
    from llm_reliability.experiments.experiment_runner import ExperimentRunner
    from llm_reliability.logging.config import LogConfig, configure_logging

    configure_logging(LogConfig(level=logging.INFO))

    data = path.read_text(encoding="utf-8")
    spec = ExperimentSpec.from_canonical_json(data)
    cache = ExperimentCache(enabled=not args.no_cache)
    cache_to_pass = cache if not args.no_cache else None

    def agent_factory(aspec, config):
        return AgentFactory.create(aspec.name, config)

    runner = ExperimentRunner(spec, agent_factory=agent_factory, cache=cache_to_pass)
    status = runner.run()
    print(
        f"Experiment {status.state}: {status.completed_runs} completed, {status.failed_runs} failed"
    )


def _resume_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    from llm_reliability.agents.agent_factory import AgentFactory
    from llm_reliability.cache import ExperimentCache
    from llm_reliability.experiments.experiment_runner import ExperimentRunner
    from llm_reliability.logging.config import LogConfig, configure_logging

    configure_logging(LogConfig(level=logging.INFO))

    spec_path = output_dir / "experiment_spec.json"
    if not spec_path.exists():
        print(f"Error: experiment_spec.json not found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    from llm_reliability.experiments.experiment_models import ExperimentSpec

    spec = ExperimentSpec.from_canonical_json(spec_path.read_text(encoding="utf-8"))
    spec = spec.model_copy(update={"output_dir": str(output_dir)})
    cache = ExperimentCache(enabled=True)

    def agent_factory(aspec, config):
        return AgentFactory.create(aspec.name, config)

    runner = ExperimentRunner(spec, agent_factory=agent_factory, cache=cache)
    status = runner.resume()
    print(
        f"Resume complete: {status.state} — {status.completed_runs} completed, "
        f"{status.failed_runs} failed"
    )


def _checkpoint_status(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)
    checkpoint_path = output_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        print(f"No checkpoint found in {output_dir}")
        return
    from llm_reliability.experiments.extended_models import CheckpointState

    state = CheckpointState.from_canonical_json(checkpoint_path.read_text(encoding="utf-8"))
    print(f"Experiment: {state.experiment_id}")
    print(f"Total runs: {state.total_runs}")
    print(f"Completed: {len(state.completed_indices)}")
    print(f"Failed: {len(state.failed_indices)}")
    print(f"Skipped: {len(state.skipped_indices)}")
    print(
        f"Progress: {len(state.completed_indices) / state.total_runs * 100:.1f}%"
        if state.total_runs
        else "N/A"
    )


def _compare_experiments(args: argparse.Namespace) -> None:
    dirs = [Path(d) for d in args.directories]
    for d in dirs:
        if not d.exists():
            print(f"Error: directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    summaries = []
    for d in dirs:
        summary_path = d / "experiment_summary.json"
        if summary_path.exists():
            summaries.append((d.name, json.loads(summary_path.read_text(encoding="utf-8"))))
    if not summaries:
        print("No experiment summaries found.")
        return

    for name, summary in summaries:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        rel = summary.get("reliability", {})
        sr = summary.get("success_rate", {})
        print(f"  Mean Reliability:  {rel.get('mean', 'N/A')}")
        print(f"  Mean Success Rate: {sr.get('mean', 'N/A')}")
        print(f"  Models: {', '.join(summary.get('agents', []))}")


def _report(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    from llm_reliability.reporting.publication import \
        save_publication_artifacts

    metrics, rankings, executions = _load_artifacts(output_dir)
    if not metrics:
        print("No metrics found in output directory.")
        return

    result = save_publication_artifacts(
        experiment_id=output_dir.name,
        metrics=metrics,
        rankings=rankings,
        executions=executions,
        output_dir=output_dir / "publication",
    )
    print(f"Reports generated in {output_dir / 'publication'}:")
    for name, path in result.items():
        print(f"  {name}: {path}")


def _export(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    from llm_reliability.reporting.publication import (generate_csv,
                                                       generate_latex_table,
                                                       generate_markdown_table)

    _, rankings, _ = _load_artifacts(output_dir)
    if not rankings:
        print("No rankings found.")
        return

    export_dir = output_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ("csv", "all"):
        csv_path = generate_csv(rankings, export_dir / "rankings.csv")
        print(f"CSV: {csv_path}")
    if args.format in ("latex", "all"):
        tex = generate_latex_table(rankings)
        tex_path = export_dir / "rankings.tex"
        tex_path.write_text(tex, encoding="utf-8")
        print(f"LaTeX: {tex_path}")
    if args.format in ("md", "markdown", "all"):
        md = generate_markdown_table(rankings)
        md_path = export_dir / "rankings.md"
        md_path.write_text(md, encoding="utf-8")
        print(f"Markdown: {md_path}")


def _validate_config(args: argparse.Namespace) -> None:
    path = Path(args.config)
    if not path.exists():
        print(f"Error: config file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        from llm_reliability.experiments.experiment_models import \
            ExperimentSpec

        data = path.read_text(encoding="utf-8")
        spec = ExperimentSpec.from_canonical_json(data)
        print(f"Config valid: {spec.experiment_name} ({spec.experiment_id})")
        print(f"  Benchmarks: {len(spec.benchmarks)}")
        print(f"  Agents: {len(spec.agents)}")
        print(f"  Seeds: {len(spec.seeds)} x {spec.repetitions} repetitions")
    except Exception as e:
        print(f"Config invalid: {e}", file=sys.stderr)
        sys.exit(1)


def _discover_models(args: argparse.Namespace) -> None:
    print("Discovering available models is runtime-specific.")
    print("Connect to a runtime and query its model list, e.g.:")
    print("  ollama list        # list local Ollama models")
    print("  curl <tgi>/models  # list TGI models")
    print("  curl <vllm>/models # list vLLM models")


def _discover_runtimes(args: argparse.Namespace) -> None:
    RuntimeRegistry.discover()
    names = RuntimeRegistry.list()
    if not names:
        print("No runtimes registered.")
        return
    print("Registered runtimes:")
    for name in sorted(names):
        print(f"  {name}")


def _list_benchmarks(args: argparse.Namespace) -> None:
    BenchmarkRegistry.discover()
    names = BenchmarkRegistry.list()
    if not names:
        print("No benchmarks registered.")
        return
    print("Registered benchmarks:")
    for name in sorted(names):
        print(f"  {name}")


def _hardware_info(args: argparse.Namespace) -> None:
    from llm_reliability.utils.hardware_profile import (
        HardwareRegistry, detect_hardware_profile)

    RuntimeRegistry.discover()
    profile = detect_hardware_profile(profile_id="cli-detect")
    print("=== Hardware Profile ===")
    print(f"  Profile ID:   {profile.profile_id}")
    print(f"  OS:           {profile.os_name} {profile.os_version}")
    print(f"  CPU:          {profile.cpu_architecture} ({profile.cpu_cores_logical} logical cores)")
    print(f"  RAM:          {profile.ram_total_gb} GB total")
    if profile.gpu_name:
        print(f"  GPU:          {profile.gpu_name} ({profile.vram_total_gb} GB VRAM)")
    print(f"  Python:       {profile.python_version}")
    print(f"  Node Type:    {profile.node_type}")
    print()

    print("Named hardware profiles:")
    for pid in HardwareRegistry.list_profiles():
        p = HardwareRegistry.get(pid)
        print(f"  {pid}: {p.ram_total_gb} GB RAM, {p.gpu_name or 'No GPU'}")


def _system_info(args: argparse.Namespace) -> None:
    import platform

    print("=== System Information ===")
    print(f"  Platform:     {platform.platform()}")
    print(f"  System:       {platform.system()} {platform.release()}")
    print(f"  Machine:      {platform.machine()}")
    print(f"  Processor:    {platform.processor()}")
    print(f"  Python:       {platform.python_version()}")
    print(f"  Framework:    llm-reliability-ranking v{__version__}")
    print()
    print("Registered runtimes:")
    for name in RuntimeRegistry.list():
        print(f"  {name}")
    print()
    print("Registered benchmarks:")
    for name in BenchmarkRegistry.list():
        print(f"  {name}")


def _statistics(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    from llm_reliability.reporting.publication import \
        generate_statistics_summary

    metrics, _, _ = _load_artifacts(output_dir)
    if not metrics:
        print("No metrics found.")
        return

    stats = generate_statistics_summary(metrics)
    print(json.dumps(stats, indent=2))


def _clear_cache(args: argparse.Namespace) -> None:
    from llm_reliability.cache import ExperimentCache

    cache = ExperimentCache()
    cache.clear()
    print("Cache cleared.")


def _load_artifacts(output_dir: Path):
    """Load metrics, rankings, and executions from an output directory."""
    from llm_reliability.records.execution import ExecutionRecord
    from llm_reliability.records.metric import MetricRecord
    from llm_reliability.records.ranking import RankingRecord

    metrics_path = output_dir / "metrics.json"
    rankings_path = output_dir / "rankings.json"
    executions_path = output_dir / "executions.json"

    metrics: list[MetricRecord] = []
    rankings: list[RankingRecord] = []
    executions: list[ExecutionRecord] = []

    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = [MetricRecord.model_validate(item) for item in data]
    if rankings_path.exists():
        data = json.loads(rankings_path.read_text(encoding="utf-8"))
        rankings = [RankingRecord.model_validate(item) for item in data]
    if executions_path.exists():
        data = json.loads(executions_path.read_text(encoding="utf-8"))
        executions = [ExecutionRecord.model_validate(item) for item in data]

    return metrics, rankings, executions


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="llm-reliability",
        description="LLM Reliability Ranking — evaluate and rank LLM agents",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # run
    p_run = sub.add_parser("run", help="Run an experiment from a config file")
    p_run.add_argument("config", type=str, help="Path to experiment config JSON")
    p_run.add_argument("--no-cache", action="store_true", help="Disable experiment cache")
    p_run.set_defaults(func=_run_experiment)

    # resume
    p_resume = sub.add_parser("resume", help="Resume a checkpointed experiment")
    p_resume.add_argument("output_dir", type=str, help="Experiment output directory")
    p_resume.set_defaults(func=_resume_experiment)

    # checkpoint
    p_check = sub.add_parser("checkpoint", help="Show checkpoint status")
    p_check.add_argument("output_dir", type=str, help="Experiment output directory")
    p_check.set_defaults(func=_checkpoint_status)

    # compare
    p_comp = sub.add_parser("compare", help="Compare multiple experiment outputs")
    p_comp.add_argument("directories", type=str, nargs="+", help="Experiment output directories")
    p_comp.set_defaults(func=_compare_experiments)

    # report
    p_report = sub.add_parser("report", help="Generate publication-ready reports")
    p_report.add_argument("output_dir", type=str, help="Experiment output directory")
    p_report.set_defaults(func=_report)

    # export
    p_export = sub.add_parser("export", help="Export rankings in various formats")
    p_export.add_argument("output_dir", type=str, help="Experiment output directory")
    p_export.add_argument(
        "--format",
        type=str,
        default="all",
        choices=["csv", "latex", "md", "markdown", "all"],
        help="Export format",
    )
    p_export.set_defaults(func=_export)

    # validate
    p_val = sub.add_parser("validate", help="Validate a config file")
    p_val.add_argument("config", type=str, help="Path to experiment config JSON")
    p_val.set_defaults(func=_validate_config)

    # discover-models
    p_dm = sub.add_parser("discover-models", help="Discover available models")
    p_dm.set_defaults(func=_discover_models)

    # discover-runtimes
    p_dr = sub.add_parser("discover-runtimes", help="Discover registered runtimes")
    p_dr.set_defaults(func=_discover_runtimes)

    # list benchmarks/runtimes
    p_list = sub.add_parser("list", help="List registered items")
    p_list.add_argument(
        "what",
        type=str,
        choices=["benchmarks", "runtimes"],
        help="What to list",
    )
    p_list.set_defaults(
        func=lambda a: (_list_benchmarks(a) if a.what == "benchmarks" else _discover_runtimes(a))
    )

    # hardware-info
    p_hw = sub.add_parser("hardware-info", help="Show hardware profile information")
    p_hw.set_defaults(func=_hardware_info)

    # system-info
    p_sys = sub.add_parser("system-info", help="Show system information")
    p_sys.set_defaults(func=_system_info)

    # statistics
    p_stats = sub.add_parser("statistics", help="Show experiment statistics")
    p_stats.add_argument("output_dir", type=str, help="Experiment output directory")
    p_stats.set_defaults(func=_statistics)

    # clear-cache
    p_cache = sub.add_parser("clear-cache", help="Clear experiment cache")
    p_cache.set_defaults(func=_clear_cache)

    args = parser.parse_args(argv)

    if args.version:
        _show_version(args)
        return

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
