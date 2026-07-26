#!/usr/bin/env python3
"""
run_experiment.py — CLI entry point for running LLM reliability experiments.

Usage
-----
    # Real experiment using a spec file:
    python scripts/run_experiment.py --spec path/to/spec.json

    # Real experiment via CLI flags:
    python scripts/run_experiment.py \\
        --name   "pilot" \\
        --benchmark AgentBoard \\
        --agent    openai:gpt-4o \\
        --dataset  data/agentboard.json \\
        --seeds    42 7 \\
        --reps     3 \\
        --output   results/

    # Demo / test with mock objects (no API keys required):
    python scripts/run_experiment.py --demo

Arguments
---------
--spec        Path to an ExperimentSpec JSON file.
--name        Experiment name (when building spec from CLI flags).
--benchmark   Registered benchmark name: AgentBoard | GAIA | SWEBenchLite | mock
--agent       Agent identifier: openai | anthropic | google | deepseek | qwen | llama |
              provider:model (e.g. openai:gpt-4o) | mock
--dataset     Path to dataset JSON file (required for real benchmarks).
--seeds       One or more integer seeds.
--reps        Number of repetitions per seed (default: 1).
--llm         LLM model identifier passed to Configuration.
--parallel    Enable parallel execution.
--workers     Max parallel workers (default: 4).
--output      Output directory (default: results/).
--resume      Resume a previously interrupted experiment from its checkpoint.
--demo        Use mock agent + mock benchmark (no API keys, no dataset required).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import benchmark adapters to trigger BenchmarkRegistry.register() side-effects
import llm_reliability.benchmarks.adapters.agentboard_adapter  # noqa: F401
import llm_reliability.benchmarks.adapters.gaia_adapter  # noqa: F401
import llm_reliability.benchmarks.adapters.swebench_lite_adapter  # noqa: F401
from llm_reliability.agents.agent_factory import AgentFactory
from llm_reliability.agents.mock_agent import MockAgent
from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
from llm_reliability.configs.config import Configuration
from llm_reliability.experiments import (AgentSpec, BenchmarkSpec,
                                         ExperimentRunner, ExperimentSpec)
from llm_reliability.interfaces.agent import Agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------


def _is_orchestrator_config(raw: dict) -> bool:
    """Return True if *raw* is an orchestrator config rather than a canonical ExperimentSpec.

    A canonical ExperimentSpec always contains the required field ``experiment_name``.
    An orchestrator config uses ``name`` and ``models`` (or a string-list ``benchmarks``).
    If ``experiment_name`` is absent the file cannot be a valid ExperimentSpec and
    must be routed through ExperimentOrchestrator.
    """
    return "experiment_name" not in raw


# ---------------------------------------------------------------------------
# Agent factory — resolves name → Agent instance
# ---------------------------------------------------------------------------


def _real_agent_factory(aspec: AgentSpec, config: Configuration) -> Agent:
    """Instantiate an Agent from AgentSpec using AgentFactory.

    Falls back to MockAgent only when the name is explicitly 'mock' or 'mock_agent'.
    Raises ValueError for any unrecognised name.
    """
    return AgentFactory.create(aspec.name, config)


def _demo_agent_factory(aspec: AgentSpec, config: Configuration) -> Agent:
    """Demo mode: always return MockAgent regardless of name."""
    return MockAgent(config=config)


# ---------------------------------------------------------------------------
# Benchmark factory — resolves name → Benchmark instance
# ---------------------------------------------------------------------------


def _real_benchmark_factory(name: str, config: Configuration):
    """Instantiate a real benchmark from BenchmarkRegistry.

    Falls back to MockBenchmark only when name == 'mock'.
    """
    name_lower = name.lower()
    if name_lower in ("mock", "mock_benchmark"):
        return MockBenchmark(config=config)

    from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry

    adapter_cls = BenchmarkRegistry.get(name)
    return adapter_cls(config=config)


def _demo_benchmark_factory(name: str, config: Configuration):
    """Demo mode: always return MockBenchmark regardless of name."""
    return MockBenchmark(config=config)


# ---------------------------------------------------------------------------
# Orchestrator routing
# ---------------------------------------------------------------------------


def _run_via_orchestrator(args: argparse.Namespace, is_demo: bool) -> int:
    """Route an orchestrator-format config file through ExperimentOrchestrator.

    This preserves all orchestration logic (matrix expansion, retry, multi-spec
    batch runs) without duplicating it in run_experiment.py.
    """
    from llm_reliability.orchestration.experiment_orchestrator import \
        ExperimentOrchestrator

    raw = json.loads(args.spec.read_text(encoding="utf-8"))
    output_dir = raw.get("output_dir", args.output)

    agent_factory = _demo_agent_factory if is_demo else None
    bench_factory = _demo_benchmark_factory if is_demo else None

    orchestrator = ExperimentOrchestrator(
        output_dir=output_dir,
        agent_factory=agent_factory,
        benchmark_factory=bench_factory,
    )

    if is_demo:
        log.info("Demo mode: orchestrator will use MockBenchmark + MockAgent.")

    result = orchestrator.run_from_file(args.spec, resume=args.resume)

    log.info(
        "Orchestration complete: total=%d, completed=%d, failed=%d",
        result["total_experiments"],
        result["completed_count"],
        result["failed_count"],
    )
    return 0 if result["failed_count"] == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_experiment",
        description="LLM Reliability Ranking — Experiment Runner CLI",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--spec", type=Path, help="Path to ExperimentSpec JSON file.")

    parser.add_argument("--name", default="cli_experiment", help="Experiment name.")
    parser.add_argument(
        "--benchmark",
        default="mock",
        help="Benchmark name: AgentBoard | GAIA | SWEBenchLite | mock",
    )
    parser.add_argument(
        "--agent",
        default="mock",
        help="Agent name: openai | anthropic | google | deepseek | qwen | llama | "
        "provider:model | mock",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Path to dataset JSON file (required for real benchmarks).",
    )
    parser.add_argument("--llm", default="", help="LLM model identifier (e.g. gpt-4o).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Seeds.")
    parser.add_argument("--reps", type=int, default=1, help="Repetitions per seed.")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel execution.")
    parser.add_argument("--workers", type=int, default=4, help="Max parallel workers.")
    parser.add_argument("--output", default="results", help="Output directory.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use mock agent + benchmark (no API keys or dataset required).",
    )
    return parser


def load_or_build_spec(args: argparse.Namespace) -> ExperimentSpec:
    """Load spec from file or build from CLI args."""
    if args.spec:
        log.info("Loading ExperimentSpec from %s", args.spec)
        return ExperimentSpec.from_canonical_json(args.spec.read_text(encoding="utf-8"))

    log.info("Building ExperimentSpec from CLI arguments.")

    is_demo = args.demo or args.benchmark.lower() == "mock"
    dataset_path = args.dataset if args.dataset else ("data/mock.json" if is_demo else "")

    if not is_demo and not dataset_path:
        log.warning(
            "No --dataset provided for benchmark '%s'. "
            "The benchmark will fail to load tasks unless dataset_path is set.",
            args.benchmark,
        )

    llm = (
        args.llm
        if args.llm
        else (args.agent if ":" not in args.agent else args.agent.split(":")[1])
    )

    return ExperimentSpec(
        experiment_name=args.name,
        benchmarks=[BenchmarkSpec(name=args.benchmark, dataset_path=dataset_path)],
        agents=[AgentSpec(name=args.agent)],
        seeds=args.seeds,
        repetitions=args.reps,
        parallel=args.parallel,
        max_workers=args.workers,
        output_dir=args.output,
        llm=llm or "mock",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    is_demo = args.demo or (
        args.spec is None
        and args.benchmark.lower() in ("mock", "mock_benchmark")
        and args.agent.lower() in ("mock", "mock_agent")
    )

    # ------------------------------------------------------------------
    # Schema detection: route orchestrator configs without breaking the
    # existing canonical-ExperimentSpec path.
    # ------------------------------------------------------------------
    if args.spec and args.spec.exists():
        try:
            raw = json.loads(args.spec.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            log.error("Failed to parse spec file as JSON: %s", exc)
            return 1

        if _is_orchestrator_config(raw):
            log.info(
                "Detected orchestrator config format in '%s'. "
                "Routing through ExperimentOrchestrator.",
                args.spec,
            )
            return _run_via_orchestrator(args, is_demo)

    # ------------------------------------------------------------------
    # Canonical ExperimentSpec path (existing behaviour, unchanged)
    # ------------------------------------------------------------------
    spec = load_or_build_spec(args)

    log.info(
        "Starting experiment '%s' (id=%s) | %d benchmark(s), %d agent(s), %d seed(s), %d rep(s).",
        spec.experiment_name,
        spec.experiment_id,
        len(spec.benchmarks),
        len(spec.agents),
        len(spec.seeds),
        spec.repetitions,
    )

    if is_demo:
        log.info("Demo mode: using MockBenchmark + MockAgent (no API keys required).")
        agent_factory = _demo_agent_factory
        benchmark_factory = _demo_benchmark_factory
    else:
        log.info(
            "Real mode: benchmark=%s, agent=%s",
            spec.benchmarks[0].name,
            spec.agents[0].name,
        )
        agent_factory = _real_agent_factory
        benchmark_factory = _real_benchmark_factory

    runner = ExperimentRunner(
        spec=spec,
        agent_factory=agent_factory,
        benchmark_factory=benchmark_factory,
    )

    if args.resume:
        status = runner.resume()
    else:
        status = runner.run()

    log.info(
        "Experiment finished: state=%s, completed=%d, failed=%d",
        status.state,
        status.completed_runs,
        status.failed_runs,
    )
    log.info("Results saved to: %s/", runner._result_manager.experiment_dir)
    return 0 if status.failed_runs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
