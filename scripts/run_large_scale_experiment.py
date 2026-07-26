#!/usr/bin/env python3
"""
run_large_scale_experiment.py — Large-Scale Production Experiment Runner.

Executes multi-benchmark (12 benchmarks), multi-runtime (4 runtimes),
multi-seed (5 seeds), and multi-quantization experiment specifications.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Import all 12 benchmark adapters to register them
import llm_reliability.benchmarks.adapters  # noqa: F401
from llm_reliability.agents.agent_factory import AgentFactory
from llm_reliability.agents.mock_agent import MockAgent
from llm_reliability.benchmarks.dataset_manager import DatasetManager
from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
from llm_reliability.configs.config import Configuration
from llm_reliability.experiments import (AgentSpec, BenchmarkSpec,
                                         ExperimentRunner, ExperimentSpec)
from llm_reliability.experiments.experiment_models import \
    AgentSpec as _AgentSpec
from llm_reliability.interfaces.agent import Agent
from llm_reliability.utils.hardware_profile import detect_hardware_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_large_scale_experiment")


def _real_agent_factory(aspec: _AgentSpec, config: Configuration) -> Agent:
    return AgentFactory.create(aspec.name, config)


def _demo_agent_factory(aspec: _AgentSpec, config: Configuration) -> Agent:
    return MockAgent(config=config)


def _real_benchmark_factory(name: str, config: Configuration):
    name_lower = name.lower()
    if name_lower in ("mock", "mock_benchmark"):
        return MockBenchmark(config=config)
    from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry

    adapter_cls = BenchmarkRegistry.get(name)
    return adapter_cls(config=config)


def _demo_benchmark_factory(name: str, config: Configuration):
    return MockBenchmark(config=config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_large_scale_experiment",
        description="LLM Reliability Ranking — Large Scale Production Experiment Runner",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=_REPO_ROOT / "configs" / "full_experiment_config.json",
        help="Path to full experiment specification file (JSON).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_REPO_ROOT / "results" / "full_study",
        help="Directory to save experiment outputs and checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted experiment run from checkpoint.",
    )
    parser.add_argument(
        "--download-datasets",
        action="store_true",
        help="Download benchmark datasets if missing before running.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use mock agent/benchmark factories for dry-run simulation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.info("Loading large-scale experiment configuration from: %s", args.config)

    if not args.config.exists():
        logger.error("Configuration file not found: %s", args.config)
        return 1

    with open(args.config, encoding="utf-8") as f:
        raw_config = json.load(f)

    # Detect physical hardware profile for execution manifest provenance
    hw_profile = detect_hardware_profile()
    logger.info(
        "Hardware Profile Detected: %s (CPU: %s, RAM: %.2f GB, GPU: %s)",
        hw_profile.profile_id,
        hw_profile.cpu_architecture,
        hw_profile.ram_total_gb,
        hw_profile.gpu_name or "None",
    )

    benchmarks_raw = raw_config.get(
        "benchmarks",
        [
            "GAIA",
            "MMLU",
            "HellaSwag",
            "HumanEval",
            "MBPP",
            "TruthfulQA",
            "GSM8K",
            "ARC",
            "Winogrande",
            "PIQA",
            "AgentBoard",
            "SWEBenchLite",
        ],
    )
    models_raw = raw_config.get(
        "models",
        ["ollama:llama3.1:8b", "ollama:qwen2.5:7b", "ollama:mistral:7b", "ollama:gemma2:9b"],
    )

    dataset_paths: dict[str, str] = {}
    dataset_mgr = DatasetManager(cache_dir=_REPO_ROOT / "data" / "cache")
    for bench in benchmarks_raw:
        if bench.lower() == "gaia":
            sample_file = _REPO_ROOT / "data" / "gaia_sample.json"
            dataset_paths[bench] = (
                str(sample_file) if sample_file.exists() else str(_REPO_ROOT / "data" / "GAIA")
            )
        else:
            try:
                info = dataset_mgr.get_dataset(bench)
                dataset_paths[bench] = info.file_path
            except Exception as e:
                logger.warning("Using fallback cache path for %s: %s", bench, e)
                dataset_paths[bench] = str(_REPO_ROOT / "data" / "cache" / f"{bench.lower()}.json")

    bench_specs = [
        BenchmarkSpec(
            name=b,
            dataset_path=dataset_paths.get(b, f"data/cache/{b.lower()}.json"),
        )
        for b in benchmarks_raw
    ]
    sys_prompt = raw_config.get("system_prompt")
    agent_specs: list[AgentSpec] = []
    for m in models_raw:
        if isinstance(m, str) and m.startswith("ollama:"):
            provider, model_name = m.split(":", 1)
            meta = {"model": model_name}
            if sys_prompt:
                meta["system_prompt"] = sys_prompt
            agent_specs.append(AgentSpec(name=provider, metadata=meta, agent_metadata=meta))
        elif isinstance(m, dict):
            provider = m.get("provider", "ollama")
            meta = dict(m.get("metadata", {}))
            if "model" in m:
                meta["model"] = m["model"]
            if sys_prompt and "system_prompt" not in meta:
                meta["system_prompt"] = sys_prompt
            agent_specs.append(AgentSpec(name=provider, metadata=meta, agent_metadata=meta))
        else:
            meta = {"model": str(m)}
            if sys_prompt:
                meta["system_prompt"] = sys_prompt
            agent_specs.append(AgentSpec(name=str(m), metadata=meta, agent_metadata=meta))

    spec = ExperimentSpec(
        experiment_name=raw_config.get("name", "large_scale_study"),
        benchmarks=bench_specs,
        agents=agent_specs,
        seeds=raw_config.get("seeds", [42, 100, 2026, 777, 999]),
        repetitions=raw_config.get("repetitions", 3),
        parallel=raw_config.get("parallel", False),
        max_workers=raw_config.get("max_workers", 4),
        output_dir=str(args.output_dir),
        llm=models_raw[0] if models_raw else "mock",
    )

    total_runs = len(spec.benchmarks) * len(spec.agents) * len(spec.seeds) * spec.repetitions
    logger.info(
        "ExperimentSpec created. Total runs scheduled across 12 benchmarks & 5 seeds: %d",
        total_runs,
    )

    if args.demo:
        logger.info("Demo mode: using MockBenchmark + MockAgent.")
        agent_factory = _demo_agent_factory
        benchmark_factory = _demo_benchmark_factory
    else:
        logger.info("Real mode: dispatching agents via AgentFactory.")
        agent_factory = _real_agent_factory
        benchmark_factory = _real_benchmark_factory

    runner = ExperimentRunner(
        spec=spec,
        agent_factory=agent_factory,
        benchmark_factory=benchmark_factory,
    )

    start_time = time.time()
    if args.resume:
        logger.info("Resuming experiment from checkpoint...")
        status = runner.resume()
    else:
        logger.info("Starting fresh experiment execution...")
        status = runner.run()

    elapsed = time.time() - start_time
    logger.info("Large-scale experiment complete in %.2f seconds.", elapsed)
    logger.info(
        "State: %s | Completed runs: %d | Failed runs: %d",
        status.state,
        status.completed_runs,
        status.failed_runs,
    )
    return 0 if status.failed_runs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
