"""
Purpose
-------
Orchestrate the end-to-end execution of a benchmark experiment.

Responsibilities
----------------
- Load benchmark and initialize agent
- Execute all tasks in the benchmark
- Convert executions to EvaluationRecords
- Aggregate evaluation results into MetricRecords and RankingRecords
- Gracefully handle and log component errors
- Persist experiment results to disk
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field

from llm_reliability.configs.config import Configuration
from llm_reliability.interfaces.agent import Agent
from llm_reliability.interfaces.benchmark import Benchmark
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord
from llm_reliability.utils.serialization import SerializableModel

logger = logging.getLogger(__name__)


class ExperimentResult(SerializableModel):
    """Immutable collection of all artifacts produced by an experiment run."""

    configuration: Configuration
    execution_records: list[ExecutionRecord]
    evaluation_records: list[EvaluationRecord]
    metric_records: list[MetricRecord]
    ranking_records: list[RankingRecord]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentPipeline:
    """Orchestrates an end-to-end experiment run."""

    def __init__(
        self,
        config: Configuration,
        benchmark: Benchmark,
        agent: Agent,
    ) -> None:
        """Initialize the pipeline with its core components."""
        self.config = config
        self.benchmark = benchmark
        self.agent = agent

        self.execution_records: list[ExecutionRecord] = []
        self.evaluation_records: list[EvaluationRecord] = []
        self.metric_records: list[MetricRecord] = []
        self.ranking_records: list[RankingRecord] = []
        self.errors: list[dict[str, Any]] = []

    def run(self) -> ExperimentResult:
        """Execute the entire experiment pipeline end-to-end."""
        logger.info("experiment start")
        self.agent.initialize()
        try:
            self.run_all()
            self.evaluate()
            self.compute_metrics()
            self.compute_rankings()
        finally:
            self.agent.shutdown()

        logger.info("experiment completion")
        return ExperimentResult(
            configuration=self.config,
            execution_records=self.execution_records,
            evaluation_records=self.evaluation_records,
            metric_records=self.metric_records,
            ranking_records=self.ranking_records,
            metadata={"errors": self.errors},
        )

    def run_all(self) -> None:
        """Run all tasks provided by the benchmark."""
        self.benchmark.load()
        tasks = self.benchmark.list_tasks()

        for task_id in tasks:
            for _ in range(self.config.repetitions):
                try:
                    task = self.benchmark.get_task(task_id)
                    self.run_task(task)
                except Exception as e:
                    logger.error("errors retrieving task %s: %s", task_id, e, exc_info=True)
                    self.errors.append({"phase": "run_all", "task_id": task_id, "error": str(e)})

    def run_task(self, task: dict[str, Any]) -> None:
        """Execute a single task and store the ExecutionRecord."""
        task_id = task.get("task_id", "unknown")
        logger.info("task start: %s", task_id)

        use_perturbations = bool(self.config.perturbations)
        use_faults = bool(self.config.fault_injection)

        if not use_perturbations and not use_faults:
            self.agent.reset()
            try:
                execution = self.benchmark.run(self.agent, task)
                self.execution_records.append(execution)
                logger.info("task completion: %s", task_id)
            except Exception as e:
                logger.error("errors executing task %s: %s", task_id, e, exc_info=True)
                self.errors.append({"phase": "run_task", "task_id": task_id, "error": str(e)})
        else:
            if use_perturbations:
                from llm_reliability.reliability.perturbation.manager import PerturbationManager
                pm = PerturbationManager(config=self.config)
                pert_res = pm.run_perturbed_task(self.agent, self.benchmark, task)
                self.execution_records.extend(pert_res.execution_records)
                if pert_res.errors:
                    self.errors.extend(pert_res.errors)

            if use_faults:
                from llm_reliability.reliability.faults.manager import FaultManager
                fm = FaultManager(config=self.config)
                fault_res = fm.run_fault_injected_task(self.agent, self.benchmark, task)
                if use_perturbations:
                    # Filter out baseline run to avoid duplicate baseline runs
                    self.execution_records.extend([r for r in fault_res.execution_records if r.fault_injected])
                else:
                    self.execution_records.extend(fault_res.execution_records)
                if fault_res.errors:
                    self.errors.extend(fault_res.errors)

    def evaluate(self) -> None:
        """Evaluate all captured execution records."""
        logger.info("evaluation start")
        for execution in self.execution_records:
            try:
                evaluation = self.benchmark.evaluate(execution)
                self.evaluation_records.append(evaluation)
            except Exception as e:
                logger.error("errors evaluating execution %s: %s", execution.task_id, e, exc_info=True)
                self.errors.append({"phase": "evaluate", "task_id": execution.task_id, "error": str(e)})

    def compute_metrics(self) -> None:
        """Compute aggregate reliability metrics from evaluation records.

        Groups evaluations by (benchmark, agent) so that MetricRecord.from_evaluations
        receives a homogeneous batch.  This is necessary because PerturbationManager
        and FaultManager may produce ExecutionRecords whose benchmark/agent fields
        differ from the logical config values.
        """
        logger.info("metric generation")
        if not self.evaluation_records:
            return

        computed_at = datetime.now(timezone.utc).isoformat()

        # Group evaluations by (benchmark, agent) — mirrors ExperimentRunner._aggregate()
        from collections import defaultdict
        groups: dict[tuple[str, str], list] = defaultdict(list)
        for ev in self.evaluation_records:
            groups[(ev.benchmark, ev.agent)].append(ev)

        for (bench, agent_name), group_evals in groups.items():
            try:
                metric = MetricRecord.from_evaluations(
                    group_evals,
                    task_id=None,
                    computed_at=computed_at,
                )
                self.metric_records.append(metric)
                logger.debug("Metric computed: benchmark=%s agent=%s", bench, agent_name)
            except Exception as e:
                logger.error(
                    "errors computing metrics for (%s, %s): %s",
                    bench, agent_name, e, exc_info=True,
                )
                self.errors.append({"phase": "compute_metrics", "benchmark": bench, "agent": agent_name, "error": str(e)})

    def compute_rankings(self) -> None:
        """Compute rankings from aggregated metric records."""
        logger.info("ranking generation")
        if not self.metric_records:
            return

        try:
            computed_at = datetime.now(timezone.utc).isoformat()
            for rtype in ["success", "reliability"]:
                ranking = RankingRecord.from_metrics(
                    self.metric_records,
                    ranking_type=rtype, # type: ignore
                    computed_at=computed_at,
                )
                self.ranking_records.append(ranking)
        except Exception as e:
            logger.error("errors computing rankings: %s", e, exc_info=True)
            self.errors.append({"phase": "compute_rankings", "error": str(e)})

    def save_results(self, path: Path | str) -> None:
        """Save the experiment result artifact to disk."""
        result = ExperimentResult(
            configuration=self.config,
            execution_records=self.execution_records,
            evaluation_records=self.evaluation_records,
            metric_records=self.metric_records,
            ranking_records=self.ranking_records,
            metadata={"errors": self.errors},
        )
        Path(path).write_text(result.canonical_json(), encoding="utf-8")

    @classmethod
    def load_results(cls, path: Path | str) -> ExperimentResult:
        """Load an experiment result artifact from disk."""
        return ExperimentResult.from_canonical_json(Path(path).read_text(encoding="utf-8"))
