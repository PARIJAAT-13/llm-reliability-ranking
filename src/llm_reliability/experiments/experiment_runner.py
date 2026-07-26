"""
Experiment Runner.

Orchestrates complete experiments across multiple benchmarks, agents, seeds,
and repeated runs.  Supports sequential and parallel execution, checkpointing,
and automatic resume.

How experiments are executed
----------------------------
1. ExperimentRunner receives an ExperimentSpec.
2. Scheduler generates the ordered list of RunDescriptors.
3. For each RunDescriptor the runner:
   a. Instantiates the benchmark adapter (via BenchmarkRegistry).
   b. Instantiates the agent (caller-supplied factory or registry).
   c. Runs ExperimentPipeline.run() to produce execution/evaluation records.
4. After all runs, MetricRecord.from_evaluations() and RankingRecord.from_metrics()
   are called to aggregate results.
5. ResultManager writes every artifact to disk and updates the checkpoint.

Deterministic execution
-----------------------
Seeds are derived per (base_seed, benchmark, agent, run_index) via SeedManager
so the order of execution cannot affect reproducibility.

Resuming experiments
---------------------
If a checkpoint exists in the output directory, run indices that already
completed are skipped.  The remaining runs proceed in the original scheduler
order.
"""

from __future__ import annotations

import concurrent.futures
import logging
from datetime import datetime, timezone
from typing import Any

from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry
from llm_reliability.cache.experiment_cache import ExperimentCache
from llm_reliability.configs.config import Configuration
from llm_reliability.experiments.experiment_models import (
    AgentSpec,
    ExperimentSpec,
    ExperimentState,
    ExperimentStatus,
)
from llm_reliability.experiments.result_manager import ResultManager
from llm_reliability.experiments.scheduler import RunDescriptor, Scheduler
from llm_reliability.experiments.utils import setup_experiment_logger
from llm_reliability.interfaces.agent import Agent
from llm_reliability.interfaces.benchmark import Benchmark
from llm_reliability.logging.config import get_logger
from llm_reliability.pipeline.experiment_pipeline import (
    ExperimentPipeline,
    ExperimentResult,
)
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord

logger = logging.getLogger(__name__)
log = get_logger(__name__)


class ExperimentRunner:
    """Primary entry point for running multi-benchmark, multi-agent experiments.

    Parameters
    ----------
    spec : ExperimentSpec
        Full experiment specification.
    agent_factory : callable[[AgentSpec, Configuration], Agent] | None
        Optional callable that creates agent instances.  If not provided,
        the runner expects a ``MockAgent`` (for tests) or raises RuntimeError.
    benchmark_factory : callable[[str, Configuration], Benchmark] | None
        Optional callable that creates benchmark instances via the registry.
        Defaults to using BenchmarkRegistry.get().
    cache : ExperimentCache | None
        Optional experiment cache.  When provided and enabled, repeated
        runs with identical configuration return cached results.
    """

    def __init__(
        self,
        spec: ExperimentSpec,
        agent_factory: Any | None = None,
        benchmark_factory: Any | None = None,
        cache: ExperimentCache | None = None,
    ) -> None:
        self._spec = spec
        self._agent_factory = agent_factory
        self._benchmark_factory = benchmark_factory or self._default_benchmark_factory
        self._cache = cache

        self._scheduler = Scheduler(spec)
        self._result_manager = ResultManager(spec, output_dir=spec.output_dir)

        log_dir = self._result_manager.experiment_dir / "logs"
        self._logger = setup_experiment_logger(log_dir, spec.experiment_id)

        self._status = ExperimentStatus(
            experiment_id=spec.experiment_id,
            total_runs=self._scheduler.total_runs(),
        )

        # Collected artifacts
        self._executions: list[ExecutionRecord] = []
        self._evaluations: list[EvaluationRecord] = []
        self._metrics: list[MetricRecord] = []
        self._rankings: list[RankingRecord] = []

        self._stopped = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> ExperimentStatus:
        """Execute all scheduled runs sequentially.

        Returns
        -------
        ExperimentStatus
            Final status after all runs complete (or experiment is stopped).
        """
        return self._execute(resume=False)

    def run_all(self) -> ExperimentStatus:
        """Alias for run() — execute all scheduled runs."""
        return self.run()

    def resume(self) -> ExperimentStatus:
        """Resume a previously interrupted experiment using the checkpoint.

        Returns
        -------
        ExperimentStatus
            Final status after resumed runs complete.
        """
        return self._execute(resume=True)

    def stop(self) -> None:
        """Signal the runner to stop after the current run completes."""
        self._stopped = True
        self._logger.warning("Stop requested — will halt after current run.")

    def status(self) -> ExperimentStatus:
        """Return the current execution status."""
        return self._status

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _execute(self, resume: bool) -> ExperimentStatus:
        """Core execution loop with optional checkpoint resume."""
        self._status.state = ExperimentState.RUNNING
        self._status.started_at = datetime.now(timezone.utc).isoformat()
        self._logger.info(
            "Experiment '%s' starting (id=%s, resume=%s, total_runs=%d).",
            self._spec.experiment_name,
            self._spec.experiment_id,
            resume,
            self._status.total_runs,
        )
        log.info(
            "Experiment started",
            extra={
                "event": "experiment_start",
                "experiment_id": self._spec.experiment_id,
                "experiment_name": self._spec.experiment_name,
                "total_runs": self._status.total_runs,
                "resume": resume,
            },
        )

        import time as _time_mod

        self._start_time = _time_mod.time()

        self._result_manager.save_configuration()

        run_queue = self._scheduler.build_run_queue()

        completed_indices: set[int] = set()
        if resume:
            completed_indices = set(self._result_manager.load_checkpoint())
            self._logger.info("Resuming: %d runs already completed.", len(completed_indices))
            log.info(
                "Experiment resumed",
                extra={
                    "event": "experiment_resume",
                    "experiment_id": self._spec.experiment_id,
                    "completed_runs": len(completed_indices),
                },
            )

        if self._spec.parallel:
            self._run_parallel(run_queue, completed_indices)
        else:
            self._run_sequential(run_queue, completed_indices)

        self._aggregate()
        self._finalize()
        return self._status

    def _run_sequential(
        self,
        run_queue: list[RunDescriptor],
        completed_indices: set[int],
    ) -> None:
        """Execute runs one at a time in scheduler order."""
        for idx, run in enumerate(run_queue):
            if self._stopped:
                self._logger.warning("Runner stopped at run index %d.", idx)
                break
            if idx in completed_indices:
                self._logger.debug("Skipping already-completed run index %d.", idx)
                self._status.completed_runs += 1
                continue
            self._execute_single_run(idx, run)

    def _run_parallel(
        self,
        run_queue: list[RunDescriptor],
        completed_indices: set[int],
    ) -> None:
        """Execute runs in parallel using a thread pool."""
        pending = [(idx, run) for idx, run in enumerate(run_queue) if idx not in completed_indices]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._spec.max_workers) as pool:
            futures = {pool.submit(self._execute_single_run, idx, run): idx for idx, run in pending}
            for future in concurrent.futures.as_completed(futures):
                if self._stopped:
                    break
                exc = future.exception()
                if exc:
                    run_idx = futures[future]
                    self._logger.error("Run %d raised exception: %s", run_idx, exc)

    def _execute_single_run(self, idx: int, run: RunDescriptor) -> None:
        """Execute a single RunDescriptor and collect its artifacts."""
        if not hasattr(self, "_start_time") or self._start_time is None:
            import time

            self._start_time = time.time()

        import time

        current_num = idx + 1
        total_runs = self._status.total_runs
        pct = (current_num / total_runs) * 100.0
        elapsed_sec = time.time() - self._start_time
        avg_run_sec = elapsed_sec / current_num if current_num > 0 else 0.0
        eta_sec = avg_run_sec * (total_runs - current_num)

        model_name = run.agent_name.split(":", 1)[1] if ":" in run.agent_name else run.agent_name

        def _fmt(seconds: float) -> str:
            s = int(round(seconds))
            m, sec = divmod(s, 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

        self._logger.info(
            "Progress: Run %d/%d (%.1f%%) | Benchmark: %s | Model: %s | Seed: %d | Elapsed: %s | ETA: %s",
            current_num,
            total_runs,
            pct,
            run.benchmark_name,
            model_name,
            run.derived_seed,
            _fmt(elapsed_sec),
            _fmt(eta_sec),
        )
        self._status.current_benchmark = run.benchmark_name
        self._status.current_agent = run.agent_name

        config = self._build_config(run)

        run_start = time.time()
        try:
            benchmark = self._benchmark_factory(run.benchmark_name, config)
            agent = self._build_agent(run.agent_name, config)

            pipeline = ExperimentPipeline(
                config=config, benchmark=benchmark, agent=agent, cache=self._cache
            )
            result: ExperimentResult = pipeline.run()

            self._executions.extend(result.execution_records)
            self._evaluations.extend(result.evaluation_records)

            self._status.completed_runs += 1
            self._result_manager.save_checkpoint(list(range(self._status.completed_runs)))

            run_duration = time.time() - run_start
            log.info(
                "Benchmark run completed",
                extra={
                    "event": "benchmark_complete",
                    "benchmark": run.benchmark_name,
                    "agent": run.agent_name,
                    "model": model_name,
                    "seed": run.derived_seed,
                    "run_index": idx,
                    "duration_seconds": round(run_duration, 3),
                    "num_executions": len(result.execution_records),
                    "num_evaluations": len(result.evaluation_records),
                },
            )

        except Exception as exc:
            run_duration = time.time() - run_start
            self._logger.error(
                "Run %d failed for model '%s': %s", current_num, model_name, exc, exc_info=True
            )
            self._status.failed_runs += 1
            self._status.errors.append(
                {
                    "run_index": idx,
                    "benchmark": run.benchmark_name,
                    "agent": run.agent_name,
                    "model": model_name,
                    "error": str(exc),
                }
            )
            log.error(
                "Benchmark run failed",
                extra={
                    "event": "benchmark_failure",
                    "benchmark": run.benchmark_name,
                    "agent": run.agent_name,
                    "model": model_name,
                    "seed": run.derived_seed,
                    "run_index": idx,
                    "duration_seconds": round(run_duration, 3),
                    "error": str(exc),
                },
            )

        self._result_manager.save_status(self._status)

    # ------------------------------------------------------------------
    # Aggregation & finalization
    # ------------------------------------------------------------------

    def _aggregate(self) -> None:
        """Compute MetricRecords and RankingRecords from collected evaluations."""
        if not self._evaluations:
            self._logger.warning("No evaluations collected; skipping aggregation.")
            return

        computed_at = datetime.now(timezone.utc).isoformat()

        # Group evaluations by (benchmark, agent)
        from collections import defaultdict

        groups: dict[tuple[str, str], list[EvaluationRecord]] = defaultdict(list)
        for ev in self._evaluations:
            groups[(ev.benchmark, ev.agent)].append(ev)

        for (bench, agent_name), evals in groups.items():
            try:
                metric = MetricRecord.from_evaluations(evals, computed_at=computed_at)
                self._metrics.append(metric)
                self._logger.debug("Metric computed: benchmark=%s agent=%s", bench, agent_name)
            except Exception as exc:
                self._logger.error(
                    "Metric computation failed for (%s, %s): %s", bench, agent_name, exc
                )

        # Build rankings per benchmark if ≥ 2 agents
        from collections import defaultdict as dd

        bench_metrics: dict[str, list[MetricRecord]] = dd(list)
        for m in self._metrics:
            bench_metrics[m.benchmark].append(m)

        for bench, metrics_list in bench_metrics.items():
            if len(metrics_list) < 2:
                self._logger.info(
                    "Skipping ranking for benchmark '%s': need ≥ 2 agents, got %d.",
                    bench,
                    len(metrics_list),
                )
                continue
            for rtype in ("success", "reliability"):
                try:
                    ranking = RankingRecord.from_metrics(
                        metrics_list,
                        ranking_type=rtype,
                        computed_at=computed_at,
                    )
                    self._rankings.append(ranking)
                except Exception as exc:
                    self._logger.error("Ranking '%s' failed for '%s': %s", rtype, bench, exc)

        self._result_manager.save_executions(self._executions)
        self._result_manager.save_evaluations(self._evaluations)
        self._result_manager.save_metrics(self._metrics)
        self._result_manager.save_rankings(self._rankings)

    def _finalize(self) -> None:
        """Mark experiment as complete and write final metadata."""
        self._status.completed_at = datetime.now(timezone.utc).isoformat()
        if self._status.failed_runs == 0 and not self._stopped:
            self._status.state = ExperimentState.COMPLETED
        elif self._stopped:
            self._status.state = ExperimentState.PAUSED
        else:
            self._status.state = ExperimentState.FAILED

        self._result_manager.save_metadata(self._status)
        self._result_manager.save_status(self._status)
        self._logger.info(
            "Experiment '%s' %s. runs: %d completed, %d failed.",
            self._spec.experiment_name,
            self._status.state,
            self._status.completed_runs,
            self._status.failed_runs,
        )

        duration = 0.0
        if hasattr(self, "_start_time") and self._start_time is not None:
            import time

            duration = time.time() - self._start_time
        log.info(
            "Experiment finished",
            extra={
                "event": "experiment_finish",
                "experiment_id": self._spec.experiment_id,
                "experiment_name": self._spec.experiment_name,
                "state": self._status.state,
                "total_runs": self._status.total_runs,
                "completed_runs": self._status.completed_runs,
                "failed_runs": self._status.failed_runs,
                "duration_seconds": round(duration, 3),
            },
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    def _find_agent_spec(self, agent_name: str) -> AgentSpec | None:
        """Locate an AgentSpec matching either exact name or composite name:model."""
        for a in self._spec.agents:
            model = a.metadata.get("model") or a.agent_metadata.get("model")
            full_name = f"{a.name}:{model}" if model and ":" not in a.name else a.name
            if a.name == agent_name or full_name == agent_name:
                return a
        return None

    def _build_config(self, run: RunDescriptor) -> Configuration:
        """Build a Configuration from a RunDescriptor."""
        aspec = self._find_agent_spec(run.agent_name)
        agent_meta = {}
        if aspec:
            if aspec.agent_metadata:
                agent_meta.update(aspec.agent_metadata)
            if aspec.metadata:
                agent_meta.update(aspec.metadata)

        merged_metadata = {**agent_meta, "dataset_path": run.dataset_path}

        return Configuration(
            experiment_name=self._spec.experiment_name,
            benchmark=run.benchmark_name,
            agent=run.agent_name,
            llm=self._spec.llm,
            prompt_version=self._spec.prompt_version,
            dataset_version=self._spec.dataset_version,
            seed=run.derived_seed,
            repetitions=1,
            perturbations=tuple(self._spec.perturbations),
            fault_injection=self._spec.fault_injection,
            metadata=merged_metadata,
        )

    @staticmethod
    def _default_benchmark_factory(name: str, config: Configuration) -> Benchmark:
        """Instantiate a benchmark adapter via BenchmarkRegistry."""
        adapter_cls = BenchmarkRegistry.get(name)
        return adapter_cls(config=config)

    def _build_agent(self, name: str, config: Configuration) -> Agent:
        """Instantiate an agent via the caller-supplied factory."""
        if self._agent_factory is None:
            raise RuntimeError(f"No agent_factory provided. Cannot instantiate agent '{name}'.")
        aspec = self._find_agent_spec(name)
        if aspec is None:
            raise RuntimeError(f"Agent spec not found for '{name}'.")
        return self._agent_factory(aspec, config)

    # ------------------------------------------------------------------
    # Properties for collected artifacts
    # ------------------------------------------------------------------

    @property
    def executions(self) -> list[ExecutionRecord]:
        return list(self._executions)

    @property
    def evaluations(self) -> list[EvaluationRecord]:
        return list(self._evaluations)

    @property
    def metrics(self) -> list[MetricRecord]:
        return list(self._metrics)

    @property
    def rankings(self) -> list[RankingRecord]:
        return list(self._rankings)
