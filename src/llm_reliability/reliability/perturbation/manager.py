"""
Purpose
-------
Orchestrate prompt perturbation strategy execution and result collection.

Responsibilities
----------------
- Manage configured perturbation strategies based on Configuration or explicit parameters.
- Generate perturbed task variants from an original task payload.
- Execute baseline and perturbed tasks using existing Benchmark and Agent interfaces.
- Produce linked ExecutionRecord and EvaluationRecord artifacts for robustness analysis.
- Handle strategy failures gracefully by logging errors and continuing remaining strategies.

Design notes
------------
PerturbationManager operates at the task level and reuses benchmark execution methods.
Leaf execution records set their ``perturbation`` field to match the strategy name,
ensuring full compatibility with downstream metrics, rankings, and serialization.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from llm_reliability.configs.config import Configuration
from llm_reliability.interfaces.agent import Agent
from llm_reliability.interfaces.benchmark import Benchmark
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.reliability.perturbation.base import (
    PerturbationRunResult,
    PerturbationStrategy,
)
from llm_reliability.reliability.perturbation.strategies import (
    FormattingPerturbationStrategy,
    InstructionReorderingPerturbationStrategy,
    PromptWrapperPerturbationStrategy,
    SynonymSubstitutionPerturbationStrategy,
    WhitespacePerturbationStrategy,
)

logger = logging.getLogger(__name__)


class PerturbationManager:
    """Orchestrates generation and execution of perturbed benchmark tasks.

    Parameters
    ----------
    config : Configuration | None
        Framework configuration object.
    strategies : list[PerturbationStrategy] | None
        Explicit list of strategies. If None, default strategies are initialized.
    max_perturbations : int | None
        Maximum number of perturbed task variants to generate per task.
    seed : int | None
        Random seed for deterministic perturbation generation.
    """

    def __init__(
        self,
        config: Configuration | None = None,
        strategies: list[PerturbationStrategy] | None = None,
        max_perturbations: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.max_perturbations = (
            max_perturbations
            if max_perturbations is not None
            else (
                config.metadata.get("max_perturbations")
                if config and "max_perturbations" in config.metadata
                else None
            )
        )
        self.seed = seed if seed is not None else (config.seed if config is not None else 0)

        # Register default available strategies
        self.all_strategies: dict[str, PerturbationStrategy] = {
            s.name: s
            for s in [
                WhitespacePerturbationStrategy(),
                FormattingPerturbationStrategy(),
                InstructionReorderingPerturbationStrategy(),
                SynonymSubstitutionPerturbationStrategy(),
                PromptWrapperPerturbationStrategy(),
            ]
        }

        # Filter enabled strategies based on explicit list or config
        if strategies is not None:
            self.enabled_strategies = strategies
        elif config is not None and config.perturbations:
            enabled: list[PerturbationStrategy] = []
            aliases = {
                "typo": "synonym",
                "rephrase": "reordering",
                "wrapper": "prompt_wrapper",
                "punctuation": "formatting",
            }
            for name in config.perturbations:
                resolved_name = aliases.get(name, name)
                if resolved_name.lower() in ("all", "*"):
                    enabled = list(self.all_strategies.values())
                    break
                elif resolved_name in self.all_strategies:
                    enabled.append(self.all_strategies[resolved_name])
                else:
                    logger.warning("Unknown perturbation strategy in configuration: '%s'", name)
            self.enabled_strategies = enabled or list(self.all_strategies.values())
        else:
            self.enabled_strategies = list(self.all_strategies.values())

    def generate_perturbations(
        self,
        task: dict[str, Any],
        max_perturbations: int | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate perturbed task dictionaries using enabled strategies.

        Parameters
        ----------
        task : dict[str, Any]
            Original task dictionary payload.
        max_perturbations : int | None
            Override maximum perturbations to generate.
        seed : int | None
            Override random seed.

        Returns
        -------
        list[dict[str, Any]]
            List of perturbed task dictionaries.
        """
        task_id = str(task.get("task_id", "unknown_task"))
        effective_seed = seed if seed is not None else self.seed
        limit = max_perturbations if max_perturbations is not None else self.max_perturbations

        logger.info(
            "Generating perturbations for task '%s' using %d strategies (seed=%d).",
            task_id,
            len(self.enabled_strategies),
            effective_seed,
        )

        perturbed_tasks: list[dict[str, Any]] = []

        for idx, strategy in enumerate(self.enabled_strategies):
            if limit is not None and len(perturbed_tasks) >= limit:
                logger.info("Reached maximum perturbation limit (%d). Stopping.", limit)
                break

            strat_seed = effective_seed + idx * 1000
            try:
                perturbed = strategy.apply(task, seed=strat_seed)
                meta = perturbed.get("metadata", {}).get("perturbation", {})
                pid = meta.get("perturbation_id", f"{strategy.name}_{strat_seed}")

                perturbed_tasks.append(perturbed)
                logger.info(
                    "Successfully applied strategy '%s' (perturbation_id=%s) to task '%s'.",
                    strategy.name,
                    pid,
                    task_id,
                )
            except Exception as exc:
                logger.error(
                    "Failed to apply strategy '%s' to task '%s': %s",
                    strategy.name,
                    task_id,
                    exc,
                    exc_info=True,
                )
                # Skip failed perturbation and continue remaining strategies

        return perturbed_tasks

    def run_perturbed_task(
        self,
        agent: Agent,
        benchmark: Benchmark,
        task: dict[str, Any],
        seed: int | None = None,
        max_perturbations: int | None = None,
    ) -> PerturbationRunResult:
        """Execute baseline task and perturbed variants, capturing records.

        Parameters
        ----------
        agent : Agent
            Agent instance to evaluate.
        benchmark : Benchmark
            Benchmark instance providing run and evaluate logic.
        task : dict[str, Any]
            Original task dictionary payload.
        seed : int | None
            Random seed for perturbations.
        max_perturbations : int | None
            Maximum perturbations cap.

        Returns
        -------
        PerturbationRunResult
            Collection of baseline and perturbed execution/evaluation records.
        """
        task_id = str(task.get("task_id", "unknown_task"))
        effective_seed = seed if seed is not None else self.seed

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
                perturbations=tuple(s.name for s in self.enabled_strategies),
                fault_injection=self.config.fault_injection,
                metadata=self.config.metadata,
            )
        else:
            effective_config = Configuration(
                experiment_name="perturbation_run",
                benchmark=benchmark.__class__.__name__,
                agent=agent.__class__.__name__,
                llm="default",
                prompt_version="1",
                dataset_version="1",
                seed=effective_seed,
                repetitions=1,
                perturbations=tuple(s.name for s in self.enabled_strategies),
            )

        execution_records: list[ExecutionRecord] = []
        evaluation_records: list[EvaluationRecord] = []
        errors: list[dict[str, Any]] = []

        # 1. Execute baseline run
        logger.info("Executing baseline run for task '%s'.", task_id)
        try:
            agent.reset()
            base_exec = benchmark.run(agent, task)
            # Ensure baseline has perturbation=None and correct task_id
            base_exec = base_exec.model_copy(update={"perturbation": None, "task_id": task_id})
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

        # 2. Generate perturbed task variants
        perturbed_tasks = self.generate_perturbations(
            task=task,
            max_perturbations=max_perturbations,
            seed=effective_seed,
        )

        # 3. Execute perturbed task variants
        for run_idx, ptask in enumerate(perturbed_tasks, start=1):
            p_meta = ptask.get("metadata", {}).get("perturbation", {})
            strategy_name = p_meta.get("strategy", "perturbed")

            logger.info(
                "Executing perturbed run %d/%d (strategy='%s') for task '%s'.",
                run_idx,
                len(perturbed_tasks),
                strategy_name,
                task_id,
            )

            t_start = time.perf_counter()
            try:
                agent.reset()
            except Exception as reset_exc:
                logger.warning("Agent reset failed before perturbed run %d: %s", run_idx, reset_exc)

            try:
                raw_exec = benchmark.run(agent, ptask)
                # Link record to original task_id and record strategy in perturbation field
                p_exec = raw_exec.model_copy(
                    update={
                        "task_id": task_id,
                        "perturbation": strategy_name,
                        "run_index": run_idx,
                    }
                )
                execution_records.append(p_exec)

                try:
                    p_eval = benchmark.evaluate(p_exec)
                    evaluation_records.append(p_eval)
                except Exception as eval_exc:
                    logger.error(
                        "Evaluation failed for perturbation '%s' on task '%s': %s",
                        strategy_name,
                        task_id,
                        eval_exc,
                    )
                    errors.append(
                        {
                            "phase": "evaluate_perturbed",
                            "task_id": task_id,
                            "strategy": strategy_name,
                            "error": str(eval_exc),
                        }
                    )

            except Exception as exec_exc:
                elapsed_sec = time.perf_counter() - t_start
                logger.error(
                    "Execution failed for perturbation '%s' on task '%s': %s",
                    strategy_name,
                    task_id,
                    exec_exc,
                )
                errors.append(
                    {
                        "phase": "run_perturbed",
                        "task_id": task_id,
                        "strategy": strategy_name,
                        "error": str(exec_exc),
                    }
                )

                # Create error ExecutionRecord for telemetry link
                error_record = ExecutionRecord(
                    configuration_hash=effective_config.sha256(),
                    seed=effective_seed,
                    benchmark=benchmark.__class__.__name__,
                    agent=agent.__class__.__name__,
                    task_id=task_id,
                    run_index=run_idx,
                    perturbation=strategy_name,
                    runtime_seconds=elapsed_sec,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    stdout="",
                    stderr=str(exec_exc),
                    status="error",
                    error=str(exec_exc),
                )
                execution_records.append(error_record)

                try:
                    p_eval = benchmark.evaluate(error_record)
                    evaluation_records.append(p_eval)
                except Exception as exc:
                    logger.warning("Error record evaluation also failed: %s", exc)

        return PerturbationRunResult(
            configuration=effective_config,
            task_id=task_id,
            original_task=task,
            perturbed_tasks=perturbed_tasks,
            execution_records=execution_records,
            evaluation_records=evaluation_records,
            errors=errors,
        )
