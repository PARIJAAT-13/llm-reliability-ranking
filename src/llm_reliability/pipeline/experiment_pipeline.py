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

from llm_reliability.cache.experiment_cache import ExperimentCache
from llm_reliability.configs.config import Configuration
from llm_reliability.interfaces.agent import Agent
from llm_reliability.interfaces.benchmark import Benchmark
from llm_reliability.logging.config import get_logger
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord
from llm_reliability.utils.serialization import SerializableModel

logger = logging.getLogger(__name__)
log = get_logger(__name__)


class ExperimentResult(SerializableModel):
    """Immutable collection of all artifacts produced by an experiment run."""

    configuration: Configuration
    execution_records: list[ExecutionRecord]
    evaluation_records: list[EvaluationRecord]
    metric_records: list[MetricRecord]
    ranking_records: list[RankingRecord]
    metadata: dict[str, Any] = Field(default_factory=dict)


def classify_failure_reason(error_str: str | None) -> str:
    """Classify error message string into standard failure reason category."""
    if not error_str:
        return "none"
    err = str(error_str).lower()
    if any(term in err for term in ("memory", "ram", "vram", "alloc", "gib", "mib", "out of memory", "insufficient")):
        return "memory"
    if any(term in err for term in ("not found", "unavailable", "installed", "does not exist")):
        return "model_unavailable"
    if "timeout" in err:
        return "timeout"
    if any(term in err for term in ("connection", "network", "refused", "reset", "closed")):
        return "network"
    return "inference"


class ExperimentPipeline:
    """Orchestrates an end-to-end experiment run."""

    def __init__(
        self,
        config: Configuration,
        benchmark: Benchmark,
        agent: Agent,
        cache: ExperimentCache | None = None,
    ) -> None:
        """Initialize the pipeline with its core components.

        Parameters
        ----------
        config : Configuration
            Experiment configuration.
        benchmark : Benchmark
            Benchmark adapter instance.
        agent : Agent
            Agent instance.
        cache : ExperimentCache | None
            Optional experiment cache.  When provided and enabled, pipeline
            results are cached and reused on repeated identical executions.
        """
        self.config = config
        self.benchmark = benchmark
        self.agent = agent
        self.cache = cache

        self.execution_records: list[ExecutionRecord] = []
        self.evaluation_records: list[EvaluationRecord] = []
        self.metric_records: list[MetricRecord] = []
        self.ranking_records: list[RankingRecord] = []
        self.errors: list[dict[str, Any]] = []

    def run(self) -> ExperimentResult:
        """Execute the entire experiment pipeline end-to-end.

        If a cache is configured and contains a result for this configuration,
        the cached result is returned without re-execution.
        """
        if self.cache is not None:
            key = self.cache.generate_key(self.config)
            if self.cache.exists(key):
                cached = self.cache.get(key)
                if cached is not None:
                    logger.info("Cache HIT — returning cached result for key '%s'", key)
                    log.info("Cache hit for experiment pipeline",
                             extra={"event": "cache_hit",
                                    "cache_key": key,
                                    "benchmark": self.config.benchmark,
                                    "agent": self.config.agent,
                                    "seed": self.config.seed})
                    return cached

        pipeline_start = __import__("time").time()
        log.info("Pipeline execution started",
                 extra={"event": "pipeline_start",
                        "benchmark": self.config.benchmark,
                        "agent": self.config.agent,
                        "seed": self.config.seed})

        logger.info("experiment start")
        try:
            self.agent.initialize()
        except Exception as init_exc:
            reason = classify_failure_reason(str(init_exc))
            self._log_model_skipped(self.config.agent, reason, str(init_exc))
            self._handle_unrecoverable_model_failure(reason, str(init_exc))
            self.evaluate()
            self.compute_metrics()
            self.compute_rankings()
            result = ExperimentResult(
                configuration=self.config,
                execution_records=self.execution_records,
                evaluation_records=self.evaluation_records,
                metric_records=self.metric_records,
                ranking_records=self.ranking_records,
                metadata={"errors": [{"phase": "initialize", "error": str(init_exc)}]},
            )
            self._maybe_cache(result)
            return result

        try:
            self.run_all()
            self.evaluate()
            self.compute_metrics()
            self.compute_rankings()
        finally:
            try:
                self.agent.shutdown()
            except Exception as shutdown_exc:
                logger.debug("Error during agent shutdown: %s", shutdown_exc)

        logger.info("experiment completion")
        result = ExperimentResult(
            configuration=self.config,
            execution_records=self.execution_records,
            evaluation_records=self.evaluation_records,
            metric_records=self.metric_records,
            ranking_records=self.ranking_records,
            metadata={"errors": self.errors},
        )
        self._maybe_cache(result)

        duration = __import__("time").time() - pipeline_start
        log.info("Pipeline execution completed",
                 extra={"event": "pipeline_finish",
                        "benchmark": self.config.benchmark,
                        "agent": self.config.agent,
                        "seed": self.config.seed,
                        "duration_seconds": round(duration, 3),
                        "num_executions": len(result.execution_records),
                        "num_evaluations": len(result.evaluation_records),
                        "num_errors": len(self.errors)})
        return result

    def _maybe_cache(self, result: ExperimentResult) -> None:
        """Store result in cache if caching is enabled."""
        if self.cache is not None:
            key = self.cache.generate_key(self.config)
            self.cache.set(key, result)

    def run_all(self) -> None:
        """Run all tasks provided by the benchmark with automatic model failure detection."""
        self.benchmark.load()
        tasks = self.benchmark.list_tasks()

        skipped_reason: str | None = None
        skip_error_msg: str | None = None

        for task_id in tasks:
            if skipped_reason:
                self._record_skipped_task(task_id, skipped_reason, skip_error_msg)
                continue

            for _ in range(self.config.repetitions):
                try:
                    task = self.benchmark.get_task(task_id)
                    self.run_task(task)
                    if self.execution_records:
                        last_exec = self.execution_records[-1]
                        if last_exec.task_id == task_id and last_exec.status == "error":
                            reason = classify_failure_reason(last_exec.error)
                            if reason in ("memory", "model_unavailable"):
                                skipped_reason = reason
                                skip_error_msg = last_exec.error
                                self._log_model_skipped(self.config.agent, reason, last_exec.error)
                                break
                    if self.errors and self.errors[-1].get("phase") == "run_task":
                        last_err = str(self.errors[-1].get("error", ""))
                        reason = classify_failure_reason(last_err)
                        if reason in ("memory", "model_unavailable"):
                            skipped_reason = reason
                            skip_error_msg = last_err
                            self._log_model_skipped(self.config.agent, reason, last_err)
                            break
                except Exception as e:
                    reason = classify_failure_reason(str(e))
                    logger.error("errors executing task %s: %s", task_id, e, exc_info=True)
                    self.errors.append({"phase": "run_all", "task_id": task_id, "error": str(e)})
                    if reason in ("memory", "model_unavailable"):
                        skipped_reason = reason
                        skip_error_msg = str(e)
                        self._log_model_skipped(self.config.agent, reason, str(e))
                        break

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
                if execution.status == "error" and execution.error:
                    reason = classify_failure_reason(execution.error)
                    meta = dict(execution.environment_metadata or {})
                    meta["failure_reason"] = reason
                    execution = execution.model_copy(update={"environment_metadata": meta})
                self.execution_records.append(execution)
                logger.info("task completion: %s", task_id)
            except Exception as e:
                logger.error("errors executing task %s: %s", task_id, e, exc_info=True)
                self.errors.append({"phase": "run_task", "task_id": task_id, "error": str(e)})
                reason = classify_failure_reason(str(e))
                if reason in ("memory", "model_unavailable"):
                    start_time = datetime.now(timezone.utc).isoformat()
                    err_record = ExecutionRecord(
                        configuration_hash=self.config.sha256(),
                        seed=self.config.seed,
                        benchmark=self.config.benchmark,
                        agent=self.config.agent,
                        task_id=task_id,
                        run_index=0,
                        runtime_seconds=0.0,
                        timestamp=start_time,
                        stdout="",
                        stderr=str(e),
                        status="error",
                        error=str(e),
                        agent_output=None,
                        environment_metadata={"failure_reason": reason},
                    )
                    self.execution_records.append(err_record)
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

    def _log_model_skipped(self, agent_name: str, reason: str, details: str) -> None:
        model_name = self.config.metadata.get("model") or agent_name
        reason_desc = "insufficient system memory" if reason == "memory" else f"model un-executable ({reason})"
        logger.warning("\nModel %s skipped.", model_name)
        logger.warning("Reason: %s.", reason_desc)
        logger.info("Continuing with next scheduled model.\n")

    def _handle_unrecoverable_model_failure(self, reason: str, error_msg: str) -> None:
        try:
            self.benchmark.load()
            tasks = self.benchmark.list_tasks()
        except Exception:
            tasks = ["unknown_task"]
        for task_id in tasks:
            self._record_skipped_task(task_id, reason, error_msg)

    def _record_skipped_task(self, task_id: str, reason: str, error_msg: str | None) -> None:
        start_time = datetime.now(timezone.utc).isoformat()
        model_name = self.config.metadata.get("model") or self.config.agent
        err_text = f"[SKIPPED] Model {model_name} skipped ({reason}): {error_msg or 'unrecoverable model failure'}"
        exec_record = ExecutionRecord(
            configuration_hash=self.config.sha256(),
            seed=self.config.seed,
            benchmark=self.config.benchmark,
            agent=self.config.agent,
            task_id=task_id,
            run_index=0,
            runtime_seconds=0.0,
            timestamp=start_time,
            stdout="",
            stderr=err_text,
            status="error",
            error=err_text,
            agent_output=None,
            environment_metadata={"failure_reason": reason, "skipped": True},
        )
        self.execution_records.append(exec_record)

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
