"""
Purpose
-------
Orchestrate fault injection strategies during task execution and record recovery telemetry.

Responsibilities
----------------
- Discover, filter, and execute configured fault injection strategies based on Configuration.
- Apply retries and track agent recovery status ("success", "partial", "failed").
- Ensure ExecutionRecord and EvaluationRecord have fault_injected=True and fault metadata attached.
- Provide robust error handling: if a fault strategy fails unexpectedly, log it, disable the strategy for the execution, and continue.

Design notes
------------
FaultManager operates at task level and reuses standard Benchmark and Agent interfaces.
Setting fault_injected=True on generated records makes them directly compatible with
compute_fault_tolerance() metric calculations.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from typing import Any

from llm_reliability.configs.config import Configuration
from llm_reliability.interfaces.agent import Agent
from llm_reliability.interfaces.benchmark import Benchmark
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.reliability.faults.base import (
    FaultInjectionStrategy,
    FaultRunResult,
    FaultTrace,
    RecoveryStatus,
)
from llm_reliability.reliability.faults.strategies import (
    ArtificialTimeoutFaultStrategy,
    ContextTruncationFaultStrategy,
    InvalidModelResponseFaultStrategy,
    NetworkInterruptionFaultStrategy,
    TemporaryApiFailureFaultStrategy,
    ToolFailureFaultStrategy,
)

logger = logging.getLogger(__name__)


class FaultManager:
    """Orchestrates fault injection strategies and collects agent recovery telemetry.

    Parameters
    ----------
    config : Configuration | None
        Framework configuration object.
    strategies : list[FaultInjectionStrategy] | None
        Explicit list of fault strategies. If None, defaults are loaded.
    fault_probability : float | None
        Probability [0.0, 1.0] of injecting a fault when executing a task.
    max_retries : int | None
        Maximum retry attempts permitted for agent recovery.
    seed : int | None
        Random seed for deterministic fault injection.
    """

    def __init__(
        self,
        config: Configuration | None = None,
        strategies: list[FaultInjectionStrategy] | None = None,
        fault_probability: float | None = None,
        max_retries: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.fault_probability = (
            fault_probability
            if fault_probability is not None
            else (
                config.metadata.get("fault_probability", 1.0)
                if config and "fault_probability" in config.metadata
                else 1.0
            )
        )
        self.max_retries = (
            max_retries
            if max_retries is not None
            else (
                config.metadata.get("max_retries", 3)
                if config and "max_retries" in config.metadata
                else 3
            )
        )
        self.seed = seed if seed is not None else (config.seed if config is not None else 0)

        # Default registered strategies
        self.all_strategies: dict[str, FaultInjectionStrategy] = {
            s.fault_name: s
            for s in [
                ArtificialTimeoutFaultStrategy(),
                TemporaryApiFailureFaultStrategy(),
                InvalidModelResponseFaultStrategy(),
                ToolFailureFaultStrategy(),
                ContextTruncationFaultStrategy(),
                NetworkInterruptionFaultStrategy(),
            ]
        }

        # Determine enabled strategies
        if strategies is not None:
            self.enabled_strategies = strategies
        elif config is not None and "enabled_faults" in config.metadata:
            enabled: list[FaultInjectionStrategy] = []
            for name in config.metadata["enabled_faults"]:
                if name in self.all_strategies:
                    enabled.append(self.all_strategies[name])
                else:
                    logger.warning("Unknown fault strategy in configuration: '%s'", name)
            self.enabled_strategies = enabled or list(self.all_strategies.values())
        else:
            self.enabled_strategies = list(self.all_strategies.values())

        # Set of disabled strategy names due to internal errors
        self.disabled_strategies: set[str] = set()

    def run_fault_injected_task(
        self,
        agent: Agent,
        benchmark: Benchmark,
        task: dict[str, Any],
        seed: int | None = None,
    ) -> FaultRunResult:
        """Execute baseline and fault-injected runs for a single task.

        Parameters
        ----------
        agent : Agent
            LLM agent under evaluation.
        benchmark : Benchmark
            Benchmark adapter.
        task : dict[str, Any]
            Task payload dictionary.
        seed : int | None
            Random seed override.

        Returns
        -------
        FaultRunResult
            Collection of baseline and fault-injected execution and evaluation records.
        """
        task_id = str(task.get("task_id", "unknown_task"))
        effective_seed = seed if seed is not None else self.seed
        rng = random.Random(effective_seed)

        # Build effective configuration
        if self.config is not None:
            effective_config = Configuration(
                version=self.config.version,
                experiment_name=self.config.experiment_name,
                benchmark=self.config.benchmark,
                agent=self.config.agent,
                llm=self.config.llm,
                prompt_version=self.config.prompt_version,
                dataset_version=self.config.dataset_version,
                seed=effective_seed,
                repetitions=self.config.repetitions,
                perturbations=self.config.perturbations,
                fault_injection=True,
                metadata=self.config.metadata,
            )
        else:
            effective_config = Configuration(
                experiment_name="fault_run",
                benchmark=benchmark.__class__.__name__,
                agent=agent.__class__.__name__,
                llm="default",
                prompt_version="1",
                dataset_version="1",
                seed=effective_seed,
                repetitions=1,
                fault_injection=True,
            )

        execution_records: list[ExecutionRecord] = []
        evaluation_records: list[EvaluationRecord] = []
        fault_traces: list[FaultTrace] = []
        errors: list[dict[str, Any]] = []

        # 1. Execute baseline (non-faulted) run
        logger.info("Executing baseline run for task '%s'.", task_id)
        try:
            agent.reset()
            base_exec = benchmark.run(agent, task)
            base_exec = base_exec.model_copy(update={"fault_injected": False, "task_id": task_id})
            execution_records.append(base_exec)

            try:
                base_eval = benchmark.evaluate(base_exec)
                evaluation_records.append(base_eval)
            except Exception as eval_exc:
                logger.error("Baseline evaluation failed for task '%s': %s", task_id, eval_exc)
                errors.append(
                    {"phase": "evaluate_baseline", "task_id": task_id, "error": str(eval_exc)}
                )
        except Exception as exec_exc:
            logger.error("Baseline execution failed for task '%s': %s", task_id, exec_exc)
            errors.append({"phase": "run_baseline", "task_id": task_id, "error": str(exec_exc)})

        # 2. Execute fault-injected runs
        for run_idx, strategy in enumerate(self.enabled_strategies, start=1):
            if strategy.fault_name in self.disabled_strategies:
                logger.warning(
                    "Strategy '%s' is disabled due to previous errors — skipping.",
                    strategy.fault_name,
                )
                continue

            # Probability check
            if rng.random() > self.fault_probability:
                logger.info(
                    "Skipping fault '%s' based on probability threshold.", strategy.fault_name
                )
                continue

            logger.info(
                "Injecting fault '%s' (injection_point='%s') on task '%s'.",
                strategy.fault_name,
                strategy.injection_point,
                task_id,
            )
            strat_seed = effective_seed + run_idx * 500

            t_start = time.perf_counter()
            retry_count = 0
            exec_error: str | None = None
            raw_exec: ExecutionRecord | None = None
            active_task = task

            # Prepare task or environment before retry loop
            try:
                if strategy.injection_point == "prompt":
                    active_task = strategy.inject(task, seed=strat_seed)
            except Exception as strat_exc:
                logger.error(
                    "Fault strategy '%s' failed during prompt injection: %s",
                    strategy.fault_name,
                    strat_exc,
                    exc_info=True,
                )
                self.disabled_strategies.add(strategy.fault_name)
                errors.append(
                    {
                        "phase": "strategy_inject",
                        "fault_name": strategy.fault_name,
                        "error": str(strat_exc),
                    }
                )
                continue

            # Retry loop attempting execution under fault
            for attempt in range(self.max_retries + 1):
                retry_count = attempt
                try:
                    agent.reset()
                except Exception as reset_exc:
                    logger.warning("Agent reset failed before attempt %d: %s", attempt, reset_exc)

                try:
                    # Apply pre-run injection if applicable
                    if strategy.injection_point in ("agent_run", "api_call", "tool_call"):
                        strategy.inject(agent, seed=strat_seed)

                    raw_exec = benchmark.run(agent, active_task)
                    if raw_exec.status == "success" and not raw_exec.error:
                        exec_error = None
                        break
                    else:
                        exec_error = raw_exec.error
                except Exception as run_exc:
                    exec_error = str(run_exc)
                    logger.warning(
                        "Attempt %d failed for fault '%s' on task '%s': %s",
                        attempt + 1,
                        strategy.fault_name,
                        task_id,
                        run_exc,
                    )

            elapsed_sec = time.perf_counter() - t_start

            # Cleanup strategy
            try:
                strategy.cleanup()
            except Exception as clean_exc:
                logger.warning(
                    "Cleanup failed for strategy '%s': %s", strategy.fault_name, clean_exc
                )

            # Build final ExecutionRecord with fault_injected=True
            if raw_exec is not None:
                f_exec = raw_exec.model_copy(
                    update={
                        "task_id": task_id,
                        "fault_injected": True,
                        "run_index": run_idx,
                        "runtime_seconds": elapsed_sec,
                    }
                )
            else:
                # Derive names from config if available so records are consistent
                # with the rest of the experiment (avoids benchmark/agent mismatch errors downstream).
                bench_name = (
                    effective_config.benchmark
                    if self.config is not None
                    else benchmark.__class__.__name__
                )
                agent_name = (
                    effective_config.agent if self.config is not None else agent.__class__.__name__
                )
                f_exec = ExecutionRecord(
                    configuration_hash=effective_config.sha256(),
                    seed=effective_seed,
                    benchmark=bench_name,
                    agent=agent_name,
                    task_id=task_id,
                    run_index=run_idx,
                    fault_injected=True,
                    runtime_seconds=elapsed_sec,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    stdout="",
                    stderr=exec_error or "Execution error under fault",
                    status="error",
                    error=exec_error,
                )

            execution_records.append(f_exec)

            # Evaluate execution
            f_eval: EvaluationRecord | None = None
            try:
                f_eval = benchmark.evaluate(f_exec)
                evaluation_records.append(f_eval)
            except Exception as eval_exc:
                logger.error(
                    "Evaluation failed for fault '%s' on task '%s': %s",
                    strategy.fault_name,
                    task_id,
                    eval_exc,
                )
                errors.append(
                    {
                        "phase": "evaluate_faulted",
                        "task_id": task_id,
                        "fault_name": strategy.fault_name,
                        "error": str(eval_exc),
                    }
                )

            # Classify recovery status
            if f_eval is not None and f_eval.success and f_eval.score >= 1.0:
                rec_status: RecoveryStatus = "success"
            elif f_eval is not None and f_eval.score > 0.0:
                rec_status = "partial"
            else:
                rec_status = "failed"

            trace = FaultTrace(
                fault_name=strategy.fault_name,
                injection_point=strategy.injection_point,
                retry_count=retry_count,
                recovery_status=rec_status,
                execution_outcome=f_exec.status,
                latency_seconds=elapsed_sec,
                details={
                    "task_id": task_id,
                    "error": f_exec.error,
                    "score": f_eval.score if f_eval is not None else 0.0,
                },
            )
            fault_traces.append(trace)

            # Attach telemetry to ExecutionRecord environment_metadata
            f_exec.environment_metadata["fault_injection"] = trace.model_dump()

        return FaultRunResult(
            configuration=effective_config,
            task_id=task_id,
            original_task=task,
            execution_records=execution_records,
            evaluation_records=evaluation_records,
            fault_traces=fault_traces,
            errors=errors,
        )
