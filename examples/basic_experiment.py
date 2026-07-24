"""Basic experiment: run a mock benchmark with a mock agent."""

from __future__ import annotations

from llm_reliability.agents.mock_agent import MockAgent as RuntimeMock
from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
from llm_reliability.configs.config import Configuration
from llm_reliability.pipeline.experiment_pipeline import ExperimentPipeline


def main() -> None:
    config = Configuration(
        experiment_name="basic_demo",
        benchmark="MockBenchmark",
        agent="mock",
        llm="mock",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
    )

    benchmark = MockBenchmark(config=config)
    agent = RuntimeMock(config=config)

    pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=agent)
    result = pipeline.run()

    print(f"Experiment: {config.experiment_name}")
    print(f"  Executions: {len(result.execution_records)}")
    print(f"  Evaluations: {len(result.evaluation_records)}")
    print(f"  Metrics: {len(result.metric_records)}")
    print(f"  Rankings: {len(result.ranking_records)}")

    for m in result.metric_records:
        print(f"\nMetrics [{m.benchmark} / {m.agent}]:")
        print(f"  Success Rate:               {m.success_rate:.2%}")
        print(f"  Repeated-Run Consistency:   {m.repeated_run_consistency:.4f}")
        print(f"  Composite Reliability:      {m.composite_reliability:.4f}")

    for r in result.ranking_records:
        print(f"\nRankings ({r.ranking_type}):")
        for rank, (agent, score) in enumerate(r.rankings, start=1):
            print(f"  #{rank}  {agent:20s}  score={score:.4f}")


if __name__ == "__main__":
    main()
