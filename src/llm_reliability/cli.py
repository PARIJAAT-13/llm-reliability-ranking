"""Command-line interface for the LLM Reliability Ranking framework."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from llm_reliability import __version__
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry
from llm_reliability.runtime.registry import RuntimeRegistry

logger = logging.getLogger(__name__)


def _list_benchmarks(args: argparse.Namespace) -> None:
    """Print all registered benchmarks."""
    BenchmarkRegistry.discover()
    names = BenchmarkRegistry.list()
    if not names:
        print("No benchmarks registered.")
        return
    print("Registered benchmarks:")
    for name in sorted(names):
        print(f"  {name}")


def _list_runtimes(args: argparse.Namespace) -> None:
    """Print all registered runtimes."""
    RuntimeRegistry.discover()
    names = RuntimeRegistry.list()
    if not names:
        print("No runtimes registered.")
        return
    print("Registered runtimes:")
    for name in sorted(names):
        print(f"  {name}")


def _validate_config(args: argparse.Namespace) -> None:
    """Validate an experiment configuration JSON file."""
    path = Path(args.config)
    if not path.exists():
        print(f"Error: config file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        from llm_reliability.experiments.experiment_models import ExperimentSpec

        data = path.read_text(encoding="utf-8")
        spec = ExperimentSpec.from_canonical_json(data)
        print(f"Config valid: {spec.experiment_name} ({spec.experiment_id})")
        print(f"  Benchmarks: {len(spec.benchmarks)}")
        print(f"  Agents: {len(spec.agents)}")
        print(f"  Seeds: {len(spec.seeds)} x {spec.repetitions} repetitions")
    except Exception as e:
        print(f"Config invalid: {e}", file=sys.stderr)
        sys.exit(1)


def _clear_cache(args: argparse.Namespace) -> None:
    """Clear the experiment cache."""
    from llm_reliability.cache import ExperimentCache

    cache = ExperimentCache()
    cache.clear()
    print("Cache cleared.")


def _show_version(args: argparse.Namespace) -> None:
    """Print the framework version."""
    print(f"llm-reliability-ranking v{__version__}")


def _run_experiment(args: argparse.Namespace) -> None:
    """Run an experiment from a config JSON file."""
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


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="llm-reliability",
        description="LLM Reliability Ranking — evaluate and rank LLM agents",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    p_run = sub.add_parser("run", help="Run an experiment")
    p_run.add_argument("config", type=str, help="Path to experiment config JSON")
    p_run.add_argument("--no-cache", action="store_true", help="Disable experiment cache")
    p_run.set_defaults(func=_run_experiment)

    p_list = sub.add_parser("list", help="List registered items")
    p_list.add_argument(
        "what",
        type=str,
        choices=["benchmarks", "runtimes"],
        help="What to list",
    )
    p_list.set_defaults(
        func=lambda a: _list_benchmarks(a) if a.what == "benchmarks" else _list_runtimes(a)
    )

    p_validate = sub.add_parser("validate", help="Validate a config file")
    p_validate.add_argument("config", type=str, help="Path to experiment config JSON")
    p_validate.set_defaults(func=_validate_config)

    p_cache = sub.add_parser("clear-cache", help="Clear experiment cache")
    p_cache.set_defaults(func=_clear_cache)

    args = parser.parse_args(argv)

    if args.version or (not args.command and getattr(args, "version", False)):
        _show_version(args)
        return

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
