"""
Purpose
-------
Provide repeated-run execution capabilities to evaluate LLM repeated-run consistency
under identical task conditions.

Responsibilities
----------------
- Accept an Agent, Benchmark, task payload, repetition count, and random seed
- Execute the exact same task repeatedly while ensuring each execution remains independent
- Reset agent state between runs via ``agent.reset()``
- Assign unique repetition indices (0..N-1) to each ExecutionRecord and EvaluationRecord
- Preserve timestamps, latency, outputs, telemetry, and evaluation scores for every run
- Handle and record individual run failures gracefully without interrupting remaining runs
- Produce canonical, immutable RepeatedRunResult artifacts ready for metric calculation

Usage example
-------------
>>> from llm_reliability.configs.config import Configuration
>>> from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
>>> from llm_reliability.agents.mock_agent import MockAgent
>>> from llm_reliability.reliability.repeated_runner import RepeatedRunner
>>> cfg = Configuration(
...     experiment_name="test_repeated",
...     benchmark="MockBenchmark",
...     agent="MockAgent",
...     llm="mock",
...     prompt_version="v1",
...     dataset_version="v1",
...     seed=42,
...     repetitions=5,
... )
>>> benchmark = MockBenchmark(config=cfg)
>>> benchmark.load()
>>> agent = MockAgent(config=cfg)
>>> runner = RepeatedRunner(config=cfg, benchmark=benchmark, agent=agent)
>>> task = benchmark.get_task("mock-task-0")
>>> result = runner.run_repeated_task(agent, benchmark, task, repetitions=5)
>>> len(result.execution_records)
5

Design notes
------------
RepeatedRunner operates below ExperimentRunner and above individual Benchmark.run()
calls. It never duplicates ExperimentRunner's high-level multi-benchmark matrix
scheduling. Instead, it focuses purely on task-level repeated execution to collect
the telemetry needed for repeated-run consistency calculation.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from llm_reliability.configs.config import Configuration
from llm_reliability.interfaces.agent import Agent
from llm_reliability.interfaces.benchmark import Benchmark
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.utils.serialization import SerializableModel

logger = logging.getLogger(__name__)


class RepeatedRunResult(SerializableModel):
    """Immutable result container for repeated task executions.

    Stores all ExecutionRecord and EvaluationRecord artifacts generated
    across N repetitions of a single benchmark task.
    """

    configuration: Configuration
    task_id: str = Field(min_length=1)
    repetitions: int = Field(gt=0)
    execution_records: list[ExecutionRecord] = Field(default_factory=list)
    evaluation_records: list[EvaluationRecord] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def success_count(self) -> int:
        """Return number of successful evaluation records."""
        return sum(1 for ev in self.evaluation_records if ev.success)

    @property
    def failure_count(self) -> int:
        """Return number of failed or errored runs."""
        return len(self.errors) + sum(1 for ev in self.evaluation_records if not ev.success)


class RepeatedRunner:
    """Orchestrates repeated executions of a single task under identical conditions.

    Parameters
    ----------
    config : Configuration | None
        Framework configuration object.
    benchmark : Benchmark | None
        Benchmark adapter instance.
    agent : Agent | None
        LLM agent instance.
    repetitions : int | None
        Default number of repetitions (overrides config.repetitions if set).
    seed : int | None
        Default random seed (overrides config.seed if set).
    """

    def __init__(
        self,
        config: Configuration | None = None,
        benchmark: Benchmark | None = None,
        agent: Agent | None = None,
        repetitions: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.benchmark = benchmark
        self.agent = agent
        self.repetitions = repetitions
        self.seed = seed

    def run_repeated_task(
        self,
        agent: Agent | None = None,
        benchmark: Benchmark | None = None,
        task: dict[str, Any] | None = None,
        repetitions: int | None = None,
        seed: int | None = None,
    ) -> RepeatedRunResult:
        """Execute a single task repeatedly for N independent runs.

        Parameters
        ----------
        agent : Agent | None
            Agent instance to run (defaults to self.agent).
        benchmark : Benchmark | None
            Benchmark instance to run (defaults to self.benchmark).
        task : dict[str, Any] | None
            Task payload dict.
        repetitions : int | None
            Number of repetitions to execute.
        seed : int | None
            Seed value to record in configuration.

        Returns
        -------
        RepeatedRunResult
            Immutable collection of execution and evaluation records.

        Raises
        ------
        ValueError
            If required components (agent, benchmark, task) are missing.
        """
        active_agent = agent or self.agent
        active_benchmark = benchmark or self.benchmark
        active_task = task

        if active_agent is None:
            raise ValueError("An Agent instance must be provided.")
        if active_benchmark is None:
            raise ValueError("A Benchmark instance must be provided.")
        if active_task is None:
            raise ValueError("A task dictionary must be provided.")

        task_id = str(active_task.get("task_id", "unknown"))

        # Determine repetitions and seed
        num_reps = (
            repetitions
            if repetitions is not None
            else (
                self.repetitions
                if self.repetitions is not None
                else (self.config.repetitions if self.config is not None else 1)
            )
        )

        active_seed = (
            seed
            if seed is not None
            else (
                self.seed
                if self.seed is not None
                else (self.config.seed if self.config is not None else 0)
            )
        )

        # Build or update configuration
        if self.config is not None:
            effective_config = Configuration(
                version=self.config.version,
                experiment_name=self.config.experiment_name,
                benchmark=self.config.benchmark,
                agent=self.config.agent,
                llm=self.config.llm,
                prompt_version=self.config.prompt_version,
                dataset_version=self.config.dataset_version,
                seed=active_seed,
                repetitions=num_reps,
                perturbations=self.config.perturbations,
                fault_injection=self.config.fault_injection,
                metadata=self.config.metadata,
            )
        else:
            effective_config = Configuration(
                experiment_name="repeated_run",
                benchmark=active_benchmark.__class__.__name__,
                agent=active_agent.__class__.__name__,
                llm="default",
                prompt_version="1",
                dataset_version="1",
                seed=active_seed,
                repetitions=num_reps,
            )

        logger.info(
            "Starting repeated run for task '%s': repetitions=%d, seed=%d.",
            task_id,
            num_reps,
            active_seed,
        )

        execution_records: list[ExecutionRecord] = []
        evaluation_records: list[EvaluationRecord] = []
        errors: list[dict[str, Any]] = []

        for run_idx in range(num_reps):
            logger.info("Run %d/%d", run_idx + 1, num_reps)
            t_start = time.perf_counter()

            # 1. Reset agent state between runs when supported
            try:
                active_agent.reset()
            except Exception as exc:
                logger.warning("Agent reset failed on run %d/%d: %s", run_idx + 1, num_reps, exc)

            # 2. Execute task via benchmark.run()
            try:
                raw_execution = active_benchmark.run(active_agent, active_task)
                # Assign current repetition run_index to guarantee record uniqueness
                execution = raw_execution.model_copy(update={"run_index": run_idx})
                execution_records.append(execution)

                elapsed_sec = time.perf_counter() - t_start
                logger.info(
                    "Run %d/%d completed in %.3fs for task '%s'. status=%s",
                    run_idx + 1,
                    num_reps,
                    execution.runtime_seconds or elapsed_sec,
                    task_id,
                    execution.status,
                )

                # Check if benchmark returned an error execution record
                if execution.status == "error" or execution.error:
                    logger.warning(
                        "Run %d/%d recorded error: %s", run_idx + 1, num_reps, execution.error
                    )
                    errors.append(
                        {
                            "run_index": run_idx,
                            "phase": "run",
                            "task_id": task_id,
                            "error": execution.error or "Task execution error",
                        }
                    )

                # 3. Evaluate execution via benchmark.evaluate()
                try:
                    evaluation = active_benchmark.evaluate(execution)
                    evaluation_records.append(evaluation)
                except Exception as eval_exc:
                    logger.error(
                        "Evaluation failed on Run %d/%d for task '%s': %s",
                        run_idx + 1,
                        num_reps,
                        task_id,
                        eval_exc,
                        exc_info=True,
                    )
                    errors.append(
                        {
                            "run_index": run_idx,
                            "phase": "evaluate",
                            "task_id": task_id,
                            "error": str(eval_exc),
                        }
                    )

            except Exception as run_exc:
                elapsed_sec = time.perf_counter() - t_start
                logger.error(
                    "Execution failed on Run %d/%d for task '%s': %s",
                    run_idx + 1,
                    num_reps,
                    task_id,
                    run_exc,
                    exc_info=True,
                )
                errors.append(
                    {
                        "run_index": run_idx,
                        "phase": "run",
                        "task_id": task_id,
                        "error": str(run_exc),
                    }
                )

                # Create an explicit error ExecutionRecord for telemetry preservation
                error_record = ExecutionRecord(
                    configuration_hash=effective_config.sha256(),
                    seed=active_seed,
                    benchmark=active_benchmark.__class__.__name__,
                    agent=active_agent.__class__.__name__,
                    task_id=task_id,
                    run_index=run_idx,
                    runtime_seconds=elapsed_sec,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    stdout="",
                    stderr=str(run_exc),
                    status="error",
                    error=str(run_exc),
                    agent_output=None,
                )
                execution_records.append(error_record)

                # Evaluate error record
                try:
                    eval_record = active_benchmark.evaluate(error_record)
                    evaluation_records.append(eval_record)
                except Exception as exc:
                    logger.warning("Error record evaluation also failed: %s", exc)

        logger.info(
            "Repeated run complete for task '%s': %d/%d runs successful.",
            task_id,
            len(evaluation_records),
            num_reps,
        )

        return RepeatedRunResult(
            configuration=effective_config,
            task_id=task_id,
            repetitions=num_reps,
            execution_records=execution_records,
            evaluation_records=evaluation_records,
            errors=errors,
        )

    def run(
        self,
        agent: Agent | None = None,
        benchmark: Benchmark | None = None,
        task: dict[str, Any] | None = None,
        repetitions: int | None = None,
        seed: int | None = None,
    ) -> RepeatedRunResult:
        """Alias for run_repeated_task."""
        return self.run_repeated_task(
            agent=agent,
            benchmark=benchmark,
            task=task,
            repetitions=repetitions,
            seed=seed,
        )
