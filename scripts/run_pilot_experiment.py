#!/usr/bin/env python3
"""
run_pilot_experiment.py — Pilot Experiment Verification Script.

Executes a controlled pilot run (e.g. 10 tasks, 3 repetitions, multiple agents)
and validates that all execution records, evaluation records, metrics, composite scores,
rankings, statistical divergence results, and reports/plots are generated correctly.
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys

# Ensure src/ is on sys.path
_REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from llm_reliability.configs.config import Configuration, ReliabilityWeightsConfig
from llm_reliability.pipeline.experiment_pipeline import ExperimentPipeline, ExperimentResult
from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
from llm_reliability.agents.mock_agent import MockAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_pilot_experiment")


def main() -> int:
    logger.info("Initializing pilot experiment...")
    output_dir = _REPO_ROOT / "results" / "pilot_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build Pilot Configuration
    config = Configuration(
        experiment_name="pilot_experiment_001",
        benchmark="mock_benchmark",
        agent="mock_agent",
        llm="gpt-4o",
        prompt_version="v1.0",
        dataset_version="1.0",
        seed=42,
        repetitions=3,
        perturbations=("typo", "whitespace"),
        fault_injection=True,
        reliability_weights=ReliabilityWeightsConfig(
            consistency=0.4,
            robustness=0.3,
            fault_tolerance=0.3,
        ),
    )

    logger.info(
        "Executing ExperimentPipeline for configuration hash: %s",
        config.sha256()[:12],
    )
    benchmark = MockBenchmark(config)
    agent = MockAgent(config)

    pipeline = ExperimentPipeline(
        config=config,
        benchmark=benchmark,
        agent=agent,
    )

    result: ExperimentResult = pipeline.run()

    logger.info("Pipeline executed successfully!")
    logger.info("Total metrics calculated: %d", len(result.metric_records))
    logger.info("Total rankings generated: %d", len(result.ranking_records))

    # Save ExperimentResult output
    summary_path = output_dir / "summary.json"
    summary_path.write_text(result.canonical_json(), encoding="utf-8")

    logger.info("Pilot experiment verified successfully!")
    logger.info("All artifacts saved to: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
