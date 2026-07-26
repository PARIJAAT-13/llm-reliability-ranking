# Developer & Extension Guide — LLM Reliability Ranking Framework

## Overview

This guide explains how to extend, configure, and integrate custom components into the **LLM Reliability Ranking Framework**.

---

## 1. Adding a Custom Benchmark Adapter

To integrate a new benchmark (e.g., HumanEval, MMLU, web navigation benchmarks):

1. Inherit from `llm_reliability.interfaces.benchmark.Benchmark`.
2. Implement the required methods: `name`, `tasks`, and `evaluate`.
3. Register the benchmark with `BenchmarkRegistry`.

```python
from typing import Sequence
from llm_reliability.interfaces.benchmark import Benchmark, Task, TaskResult
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry

class MyCustomBenchmark(Benchmark):
    def __init__(self, dataset_path: str = "default.json"):
        self._dataset_path = dataset_path

    @property
    def name(self) -> str:
        return "my_custom_benchmark"

    def get_tasks(() -> Sequence[Task]:
        return [
            Task(task_id="task_1", prompt="Solve X", reference_answer="42"),
            Task(task_id="task_2", prompt="Solve Y", reference_answer="100"),
        ]

    def evaluate(self, task: Task, agent_output: str) -> TaskResult:
        is_success = agent_output.strip() == task.reference_answer
        return TaskResult(
            task_id=task.task_id,
            success=is_success,
            score=1.0 if is_success else 0.0,
            metadata={"raw_output": agent_output},
        )

# Register for CLI / configuration discovery
BenchmarkRegistry.register("my_custom_benchmark", MyCustomBenchmark)
```

---

## 2. Adding a Custom Agent Adapter

To integrate a new LLM agent or framework wrapper (e.g., AutoGen, LangChain, CrewAI, custom API agent):

1. Inherit from `llm_reliability.interfaces.agent.Agent`.
2. Implement `name` and `solve_task`.
3. Wire into the experiment pipeline via configuration.

```python
from llm_reliability.interfaces.agent import Agent, AgentResponse
from llm_reliability.interfaces.benchmark import Task

class MyCustomAgent(Agent):
    def __init__(self, model_name: str = "gpt-4o"):
        self._model_name = model_name

    @property
    def name(self) -> str:
        return f"my_agent_{self._model_name}"

    def solve_task(self, task: Task) -> AgentResponse:
        # Call model API / multi-turn agent logic
        output = "42"  # placeholder response
        return AgentResponse(
            output=output,
            status="success",
            runtime_seconds=0.45,
            metadata={"model": self._model_name},
        )
```

---

## 3. Running an End-to-End Experiment Pipeline

```python
from llm_reliability.configs import Configuration, ReliabilityWeightsConfig
from llm_reliability.pipeline.experiment_pipeline import ExperimentPipeline

# 1. Define configuration with optional weights
config = Configuration(
    experiment_name="pilot_eval",
    benchmark="mock_benchmark",
    agent="mock_agent",
    llm="gpt-4o",
    prompt_version="v1.0",
    dataset_version="1.0",
    seed=42,
    repetitions=5,
    reliability_weights=ReliabilityWeightsConfig(
        consistency=0.4,
        robustness=0.3,
        fault_tolerance=0.3,
    )
)

# 2. Instantiate and run pipeline
pipeline = ExperimentPipeline(config)
results = pipeline.run()

# 3. Access summary and metrics
print(f"Success Rate: {results['summary'].metrics[0].success_rate:.2%}")
```

---

## 4. Measuring Ranking Divergence

To compare conventional success rankings against reliability rankings:

```python
from llm_reliability.ranking.ranking_engine import RankingEngine
from llm_reliability.statistics.ranking_divergence import analyze_ranking_divergence

# Given a list of MetricRecords from your benchmark evaluations:
engine = RankingEngine(metrics=metric_records)
success_ranking = engine.rank_success()
reliability_ranking = engine.rank_reliability()

# Analyze divergence
divergence = analyze_ranking_divergence(success_ranking, reliability_ranking)

print(f"Overlap: {divergence.overlap:.2%}")
print(f"Divergence: {divergence.divergence:.2%}")
print(f"Mean Rank Displacement: {divergence.mean_displacement:.2f}")
```

---

## 5. Statistical Rigor & Reproducibility Guidelines

- Always specify an explicit `seed` in your `Configuration`.
- Store canonical JSON exports of `ExecutionRecord` and `EvaluationRecord` alongside experiment artifacts.
- Use `EnvironmentCapture` (from `llm_reliability.reproducibility.environment`) to capture python version, package dependencies, git commit hash, and CUDA driver versions for publication appendices.
- Avoid modifying task definitions directly in benchmark adapters; use `PerturbationManager` to apply reproducible transformations.
