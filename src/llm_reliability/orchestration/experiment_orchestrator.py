"""
Purpose
-------
Provide a declarative Experiment Orchestrator for the LLM Reliability Ranking framework.

Responsibilities
----------------
- Parse declarative experiment definition files (YAML and JSON)
- Automatically expand high-level matrix specifications (models x benchmarks x seeds)
  into concrete ExperimentSpec objects
- Batch-execute experiments using the framework's existing ExperimentRunner
- Handle failures gracefully by logging errors, updating failure reports, and
  continuing execution of remaining experiments
- Support checkpointing and automatic resumption of interrupted batch runs
- Calculate real-time progress and estimated remaining time (ETA)
- Aggregate results into master JSON/Markdown summary reports and runtime statistics

Usage example
-------------
>>> from llm_reliability.orchestration import ExperimentOrchestrator
>>> orchestrator = ExperimentOrchestrator(output_dir="results")
>>> batch_result = orchestrator.run_from_file("configs/orchestration_example.yaml")
>>> print(f"Completed: {len(batch_result['completed_experiments'])}")

Design notes
------------
The Orchestrator operates at a higher abstraction layer than ExperimentRunner.
It NEVER duplicates scheduling, execution, or metric calculation logic. Instead,
it generates valid ExperimentSpec contracts and delegates execution to
ExperimentRunner instances.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_reliability.agents.agent_factory import AgentFactory
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry
from llm_reliability.configs.config import Configuration
from llm_reliability.experiments.experiment_models import (
    AgentSpec,
    BenchmarkSpec,
    ExperimentSpec,
)
from llm_reliability.experiments.experiment_runner import ExperimentRunner
from llm_reliability.interfaces.agent import Agent
from llm_reliability.reporting.summary import ExperimentSummary

logger = logging.getLogger(__name__)

# Default dataset paths for standard benchmarks when not specified in config
DEFAULT_DATASET_PATHS: dict[str, str] = {
    "AgentBoard": "data/agentboard.json",
    "agentboard": "data/agentboard.json",
    "GAIA": "data/gaia.json",
    "gaia": "data/gaia.json",
    "SWE-bench Lite": "data/swebench.json",
    "SWEBenchLite": "data/swebench.json",
    "swebench": "data/swebench.json",
    "MockBenchmark": "data/mock.json",
    "mock": "data/mock.json",
}


class ExperimentOrchestrator:
    """Declarative batch orchestrator for multi-model, multi-benchmark studies.

    Parameters
    ----------
    output_dir : str | Path
        Root directory where experiment output subdirectories and master summaries are saved.
    agent_factory : Callable[[AgentSpec, Configuration], Agent] | None
        Optional factory for instantiating Agent objects. If None, a smart default factory
        is used that supports MockAgent, GPTAgent, and registered providers.
    benchmark_factory : Callable[[str, Configuration], Any] | None
        Optional factory for instantiating Benchmark objects via BenchmarkRegistry.
    """

    def __init__(
        self,
        output_dir: str | Path = "results",
        agent_factory: Callable[[AgentSpec, Configuration], Agent] | None = None,
        benchmark_factory: Callable[[str, Configuration], Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._agent_factory = agent_factory or self._default_agent_factory
        self._benchmark_factory = benchmark_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_from_file(
        self,
        config_path: str | Path,
        resume: bool = True,
    ) -> dict[str, Any]:
        """Load experiment matrix definition from file (YAML or JSON) and run all experiments.

        Parameters
        ----------
        config_path : str | Path
            Path to YAML (.yaml, .yml) or JSON (.json) configuration file.
        resume : bool
            Whether to attempt resuming previously interrupted experiments.

        Returns
        -------
        dict[str, Any]
            Master summary dictionary containing completion status, runtime stats,
            and error reports.
        """
        raw_def = self.load_config_file(config_path)
        specs = self.generate_specs(raw_def)
        batch_name = raw_def.get("name", Path(config_path).stem)
        return self.run_all(specs, batch_name=batch_name, resume=resume)

    def run_all(
        self,
        specs: list[ExperimentSpec],
        batch_name: str = "orchestration_batch",
        resume: bool = True,
        validate: bool = True,
        check_ollama: bool = True,
    ) -> dict[str, Any]:
        """Execute a sequence of ExperimentSpec runs sequentially with failure tolerance.

        Parameters
        ----------
        specs : list[ExperimentSpec]
            List of experiment specifications to execute.
        batch_name : str
            Identifier for this orchestration run.
        resume : bool
            If True, skips already completed runs or resumes from checkpoint.
        validate : bool
            If True, performs pre-flight matrix validation before starting execution.
        check_ollama : bool
            If True, checks Ollama server reachability during pre-flight validation.

        Returns
        -------
        dict[str, Any]
            Master summary report dictionary.
        """
        if validate:
            self.validate_specs(specs, check_ollama_server=check_ollama)

        total_experiments = len(specs)
        logger.info(
            "Starting orchestration batch '%s' with %d experiment(s).",
            batch_name,
            total_experiments,
        )

        completed_experiments: list[dict[str, Any]] = []
        failed_experiments: list[dict[str, Any]] = []
        summaries: list[ExperimentSummary] = []

        batch_start_time = time.time()
        completed_runtimes: list[float] = []

        for idx, spec in enumerate(specs, start=1):
            exp_start_time = time.time()
            progress_pct = (idx / total_experiments) * 100.0

            # Calculate ETA based on average runtime of completed experiments in this batch
            if completed_runtimes:
                avg_time = sum(completed_runtimes) / len(completed_runtimes)
                remaining_exp = total_experiments - (idx - 1)
                eta_seconds = avg_time * remaining_exp
                eta_str = self._format_duration(eta_seconds)
            else:
                eta_str = "Calculating..."

            benchmarks_str = ", ".join(b.name for b in spec.benchmarks)
            agents_str = ", ".join(a.name for a in spec.agents)

            logger.info(
                "[Orchestrator] Experiment %d / %d (%.1f%%) | Name: %s | Benchmark: %s | Agent: %s | Reps: %d | ETA: %s",
                idx,
                total_experiments,
                progress_pct,
                spec.experiment_name,
                benchmarks_str,
                agents_str,
                spec.repetitions,
                eta_str,
            )

            # Check if experiment is already fully completed on disk
            exp_dir = self.output_dir / spec.experiment_id
            checkpoint_file = exp_dir / "checkpoint.json"
            status_file = exp_dir / "status.json"

            if resume and checkpoint_file.exists() and status_file.exists():
                try:
                    status_data = json.loads(status_file.read_text(encoding="utf-8"))
                    if status_data.get("state") == "completed":
                        logger.info(
                            "Skipping already completed experiment '%s' (%s).",
                            spec.experiment_name,
                            spec.experiment_id,
                        )
                        started = status_data.get("started_at")
                        completed = status_data.get("completed_at")
                        if started and completed:
                            try:
                                s = datetime.fromisoformat(started)
                                e = datetime.fromisoformat(completed)
                                runtime = (e - s).total_seconds()
                            except Exception:
                                runtime = 0.0
                        else:
                            runtime = 0.0
                        completed_experiments.append(
                            {
                                "experiment_id": spec.experiment_id,
                                "experiment_name": spec.experiment_name,
                                "benchmark": benchmarks_str,
                                "agent": agents_str,
                                "status": "completed (cached)",
                                "runtime_seconds": runtime,
                            }
                        )
                        continue
                except Exception as exc:
                    logger.warning("Failed to parse status file for resume: %s", exc)

            # Instantiate ExperimentRunner
            try:
                runner = ExperimentRunner(
                    spec=spec,
                    agent_factory=self._agent_factory,
                    benchmark_factory=self._benchmark_factory,
                )

                # Resume or Run
                if resume and checkpoint_file.exists():
                    logger.info("Resuming experiment '%s' from checkpoint.", spec.experiment_name)
                    status = runner.resume()
                else:
                    status = runner.run()

                exp_elapsed = time.time() - exp_start_time
                completed_runtimes.append(exp_elapsed)

                if status.state == "completed" or status.failed_runs == 0:
                    logger.info(
                        "Experiment '%s' completed successfully in %.2fs.",
                        spec.experiment_name,
                        exp_elapsed,
                    )
                    exp_summary = ExperimentSummary.from_experiment_runner(
                        runner=runner,
                        experiment_name=spec.experiment_name,
                    )
                    summaries.append(exp_summary)

                    completed_experiments.append(
                        {
                            "experiment_id": spec.experiment_id,
                            "experiment_name": spec.experiment_name,
                            "benchmark": benchmarks_str,
                            "agent": agents_str,
                            "total_runs": status.total_runs,
                            "completed_runs": status.completed_runs,
                            "failed_runs": status.failed_runs,
                            "status": "completed",
                            "runtime_seconds": round(exp_elapsed, 2),
                        }
                    )
                else:
                    logger.warning(
                        "Experiment '%s' finished with failed runs (%d/%d).",
                        spec.experiment_name,
                        status.failed_runs,
                        status.total_runs,
                    )
                    failed_experiments.append(
                        {
                            "experiment_id": spec.experiment_id,
                            "experiment_name": spec.experiment_name,
                            "benchmark": benchmarks_str,
                            "agent": agents_str,
                            "errors": status.errors,
                            "status": "partial_failure",
                            "runtime_seconds": round(exp_elapsed, 2),
                        }
                    )

            except Exception as exc:
                exp_elapsed = time.time() - exp_start_time
                logger.error(
                    "Experiment '%s' failed with exception: %s. Continuing remaining batch.",
                    spec.experiment_name,
                    exc,
                    exc_info=True,
                )
                failed_experiments.append(
                    {
                        "experiment_id": spec.experiment_id,
                        "experiment_name": spec.experiment_name,
                        "benchmark": benchmarks_str,
                        "agent": agents_str,
                        "error": str(exc),
                        "status": "failed",
                        "runtime_seconds": round(exp_elapsed, 2),
                    }
                )

        batch_elapsed = time.time() - batch_start_time
        avg_runtime = (
            sum(completed_runtimes) / len(completed_runtimes) if completed_runtimes else 0.0
        )

        master_summary = {
            "orchestration_name": batch_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_experiments": total_experiments,
            "completed_count": len(completed_experiments),
            "failed_count": len(failed_experiments),
            "runtime_statistics": {
                "total_runtime_seconds": round(batch_elapsed, 2),
                "total_runtime_formatted": self._format_duration(batch_elapsed),
                "average_experiment_runtime_seconds": round(avg_runtime, 2),
            },
            "completed_experiments": completed_experiments,
            "failed_experiments": failed_experiments,
        }

        # Save master summary files (JSON and Markdown)
        self._save_master_summaries(master_summary, batch_name)

        logger.info(
            "Orchestration batch '%s' completed in %s. Success: %d/%d, Failed: %d/%d.",
            batch_name,
            self._format_duration(batch_elapsed),
            len(completed_experiments),
            total_experiments,
            len(failed_experiments),
            total_experiments,
        )

        return master_summary

    # ------------------------------------------------------------------
    # Spec Generation & Parsing
    # ------------------------------------------------------------------

    @classmethod
    def load_config_file(cls, path: str | Path) -> dict[str, Any]:
        """Load experiment matrix definition from a JSON or YAML file.

        Parameters
        ----------
        path : str | Path
            Path to definition file.

        Returns
        -------
        dict[str, Any]
            Parsed dictionary structure.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        suffix = file_path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            try:
                import yaml

                parsed = yaml.safe_load(content)
            except ImportError:
                logger.warning(
                    "PyYAML package not installed. Attempting basic JSON parse for %s.",
                    file_path,
                )
                parsed = json.loads(content)
        else:
            parsed = json.loads(content)

        if not isinstance(parsed, dict):
            raise ValueError(
                f"Configuration root must be a dictionary, got {type(parsed).__name__}."
            )

        return parsed

    @classmethod
    def generate_specs(cls, definition: dict[str, Any]) -> list[ExperimentSpec]:
        """Expand a high-level matrix definition into concrete ExperimentSpec objects.

        Supports matrix mode expansion:
        - "per_pair": Generates one ExperimentSpec per (benchmark, model) pair.
        - "per_combination": Generates one ExperimentSpec per (benchmark, model, seed).
        - "single": Generates one ExperimentSpec with all benchmarks and models.

        Default mode is "per_pair".

        Parameters
        ----------
        definition : dict[str, Any]
            Parsed definition dictionary.

        Returns
        -------
        list[ExperimentSpec]
            List of validated ExperimentSpec objects.
        """
        base_name = definition.get("name", "experiment_batch")
        output_dir = definition.get("output_dir", "results")
        repetitions = int(definition.get("repetitions", 1))
        perturbations = list(definition.get("perturbations", []))
        fault_injection = bool(definition.get("fault_injection", False))
        parallel = bool(definition.get("parallel", False))
        max_workers = int(definition.get("max_workers", 4))
        llm = definition.get("llm", "mock")
        prompt_version = str(definition.get("prompt_version", "1"))
        dataset_version = str(definition.get("dataset_version", "1"))
        matrix_mode = definition.get("matrix_mode", "per_pair")

        # Parse seeds
        raw_seeds = definition.get("seeds", [0])
        if isinstance(raw_seeds, int):
            seeds = [raw_seeds]
        else:
            seeds = [int(s) for s in raw_seeds]

        # Parse benchmarks
        raw_benchmarks = definition.get("benchmarks", ["MockBenchmark"])
        bench_specs: list[BenchmarkSpec] = []
        for b in raw_benchmarks:
            if isinstance(b, str):
                path = DEFAULT_DATASET_PATHS.get(b, f"data/{b.lower()}.json")
                bench_specs.append(BenchmarkSpec(name=b, dataset_path=path))
            elif isinstance(b, dict):
                b_name = b["name"]
                b_path = b.get(
                    "dataset_path", DEFAULT_DATASET_PATHS.get(b_name, f"data/{b_name.lower()}.json")
                )
                b_meta = b.get("adapter_metadata", {})
                bench_specs.append(
                    BenchmarkSpec(name=b_name, dataset_path=b_path, adapter_metadata=b_meta)
                )

        # Parse models / agents
        raw_models = definition.get("models") or definition.get("agents") or ["MockAgent"]
        sys_prompt = definition.get("system_prompt")
        agent_specs: list[AgentSpec] = []
        seen_models: set[tuple[str, str]] = set()
        for m in raw_models:
            if isinstance(m, str):
                if m.startswith("ollama:"):
                    provider, model_name = m.split(":", 1)
                    meta = {"model": model_name}
                    if sys_prompt:
                        meta["system_prompt"] = sys_prompt
                    aspec = AgentSpec(name=provider, metadata=meta, agent_metadata=meta)
                else:
                    meta = {"model": m}
                    if sys_prompt:
                        meta["system_prompt"] = sys_prompt
                    aspec = AgentSpec(name=m, metadata=meta, agent_metadata=meta)
            elif isinstance(m, dict):
                a_name = m.get("provider") or m.get("name") or "ollama"
                a_meta = dict(m.get("metadata") or m.get("agent_metadata") or {})
                if "model" not in a_meta and "model" in m:
                    a_meta["model"] = m["model"]
                for k in ("temperature", "max_tokens", "base_url", "system_prompt"):
                    if k in m and k not in a_meta:
                        a_meta[k] = m[k]
                if sys_prompt and "system_prompt" not in a_meta:
                    a_meta["system_prompt"] = sys_prompt
                aspec = AgentSpec(name=a_name, metadata=a_meta, agent_metadata=a_meta)
            else:
                raise ValueError(f"Invalid model entry in configuration: {m!r}")

            model_key = (
                aspec.name,
                str(aspec.agent_metadata.get("model") or aspec.metadata.get("model") or ""),
            )
            if model_key in seen_models:
                logger.warning("Duplicate model specification detected in config: %s", model_key)
            seen_models.add(model_key)
            agent_specs.append(aspec)

        import uuid

        specs: list[ExperimentSpec] = []

        if matrix_mode == "single":
            spec_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{base_name}_single"))
            spec = ExperimentSpec(
                experiment_id=spec_id,
                experiment_name=base_name,
                benchmarks=bench_specs,
                agents=agent_specs,
                seeds=seeds,
                repetitions=repetitions,
                perturbations=perturbations,
                fault_injection=fault_injection,
                parallel=parallel,
                max_workers=max_workers,
                output_dir=output_dir,
                llm=llm,
                prompt_version=prompt_version,
                dataset_version=dataset_version,
            )
            specs.append(spec)

        elif matrix_mode == "per_combination":
            for bspec in bench_specs:
                for aspec in agent_specs:
                    for seed in seeds:
                        name_slug = cls._slugify(
                            f"{base_name}_{bspec.name}_{aspec.name}_seed{seed}"
                        )
                        spec_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, name_slug))
                        spec = ExperimentSpec(
                            experiment_id=spec_id,
                            experiment_name=name_slug,
                            benchmarks=[bspec],
                            agents=[aspec],
                            seeds=[seed],
                            repetitions=repetitions,
                            perturbations=perturbations,
                            fault_injection=fault_injection,
                            parallel=parallel,
                            max_workers=max_workers,
                            output_dir=output_dir,
                            llm=aspec.agent_metadata.get("model", llm),
                            prompt_version=prompt_version,
                            dataset_version=dataset_version,
                        )
                        specs.append(spec)

        else:  # "per_pair" (default)
            for bspec in bench_specs:
                for aspec in agent_specs:
                    name_slug = cls._slugify(f"{base_name}_{bspec.name}_{aspec.name}")
                    spec_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, name_slug))
                    spec = ExperimentSpec(
                        experiment_id=spec_id,
                        experiment_name=name_slug,
                        benchmarks=[bspec],
                        agents=[aspec],
                        seeds=seeds,
                        repetitions=repetitions,
                        perturbations=perturbations,
                        fault_injection=fault_injection,
                        parallel=parallel,
                        max_workers=max_workers,
                        output_dir=output_dir,
                        llm=aspec.agent_metadata.get("model", llm),
                        prompt_version=prompt_version,
                        dataset_version=dataset_version,
                    )
                    specs.append(spec)

        return specs

    def validate_specs(
        self,
        specs: list[ExperimentSpec],
        check_ollama_server: bool = True,
    ) -> None:
        """Perform pre-flight matrix validation before experiment execution begins.

        Validates:
        - Benchmarks exist in registry (or custom factory) and dataset files exist
        - Agent providers are registered in AgentFactory (or custom factory)
        - Ollama server availability and model existence (if Ollama provider used)

        Raises
        ------
        ValueError | RuntimeError
            If any validation rule fails.
        """
        errors: list[str] = []
        ollama_models_to_check: dict[str, set[str]] = {}

        registered_benchmarks = [b.lower() for b in BenchmarkRegistry.list()]
        # Add default aliases
        registered_benchmarks.extend(
            ["mockbenchmark", "mock", "agentboard", "gaia", "swebenchlite", "swe-bench lite"]
        )

        for spec in specs:
            # Benchmark validation
            for bspec in spec.benchmarks:
                if not bspec.name:
                    errors.append("Benchmark name cannot be empty.")
                elif (
                    self._benchmark_factory is None
                    and bspec.name.lower() not in registered_benchmarks
                ):
                    errors.append(
                        f"Benchmark '{bspec.name}' is not registered in BenchmarkRegistry."
                    )

                if bspec.dataset_path and self._benchmark_factory is None:
                    p = Path(bspec.dataset_path)
                    if not p.exists() and bspec.name.lower() not in ("mockbenchmark", "mock"):
                        errors.append(
                            f"Dataset file does not exist for benchmark '{bspec.name}': {bspec.dataset_path}"
                        )

            # Agent / provider validation
            is_default_factory = (
                self._agent_factory is None or self._agent_factory == self._default_agent_factory
            )
            for aspec in spec.agents:
                if (
                    is_default_factory
                    and not AgentFactory.is_mock(aspec.name)
                    and AgentFactory.resolve(aspec.name) is None
                ):
                    errors.append(
                        f"Unsupported agent provider '{aspec.name}'. Available prefixes: {AgentFactory.available_names()}"
                    )

                # Check Ollama specifics
                if aspec.name.lower() == "ollama" or (
                    isinstance(aspec.name, str) and aspec.name.lower().startswith("ollama:")
                ):
                    model = (
                        aspec.agent_metadata.get("model")
                        or aspec.metadata.get("model")
                        or (aspec.name.split(":", 1)[1] if ":" in aspec.name else "llama3.1:8b")
                    )
                    base_url = (
                        aspec.agent_metadata.get("base_url")
                        or aspec.metadata.get("base_url")
                        or "http://127.0.0.1:11434"
                    )
                    ollama_models_to_check.setdefault(base_url, set()).add(model)

        if check_ollama_server and ollama_models_to_check:
            from llm_reliability.agents.utils.ollama_utils import (
                check_ollama_server as _check_ollama,
            )
            from llm_reliability.agents.utils.ollama_utils import (
                validate_models_exist as _validate_models,
            )

            for base_url, models in ollama_models_to_check.items():
                ok, msg = _check_ollama(base_url)
                if not ok:
                    errors.append(f"Ollama server reachable check failed for '{base_url}': {msg}")
                else:
                    valid, model_errs = _validate_models(list(models), base_url=base_url)
                    if not valid:
                        errors.extend(model_errs)

        if errors:
            err_summary = "\n".join(f"  • {e}" for e in errors)
            raise ValueError(
                f"Pre-flight configuration validation failed with {len(errors)} error(s):\n{err_summary}"
            )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_agent_factory(aspec: AgentSpec, config: Configuration) -> Agent:
        """Default agent factory that instantiates agents based on spec name.

        Delegates agent creation to AgentFactory.create, constructing a Configuration
        object populated with all metadata fields (e.g. model specifications).
        Fails fast if the agent name is unrecognised.
        """
        cfg_metadata = dict(config.metadata)
        if aspec.agent_metadata:
            cfg_metadata.update(aspec.agent_metadata)

        if "model" not in cfg_metadata and not AgentFactory.is_mock(aspec.name):
            cfg_metadata["model"] = aspec.name

        cfg = Configuration(
            version=config.version,
            experiment_name=config.experiment_name,
            benchmark=config.benchmark,
            agent=aspec.name,
            llm=config.llm if config.llm != "default" else cfg_metadata.get("model", config.llm),
            prompt_version=config.prompt_version,
            dataset_version=config.dataset_version,
            seed=config.seed,
            repetitions=config.repetitions,
            perturbations=config.perturbations,
            fault_injection=config.fault_injection,
            metadata=cfg_metadata,
        )

        return AgentFactory.create(aspec.name, cfg)

    def _save_master_summaries(self, master_summary: dict[str, Any], batch_name: str) -> None:
        """Save master summary JSON and Markdown files to output_dir."""
        name_slug = self._slugify(batch_name)
        json_path = self.output_dir / f"{name_slug}_master_summary.json"
        md_path = self.output_dir / f"{name_slug}_master_summary.md"

        json_path.write_text(json.dumps(master_summary, indent=2), encoding="utf-8")

        # Build Markdown summary
        md_lines = [
            f"# Master Experiment Summary — {batch_name}",
            "",
            f"- **Generated At**: {master_summary['generated_at']}",
            f"- **Total Experiments**: {master_summary['total_experiments']}",
            f"- **Completed**: {master_summary['completed_count']}",
            f"- **Failed**: {master_summary['failed_count']}",
            f"- **Total Runtime**: {master_summary['runtime_statistics']['total_runtime_formatted']}",
            f"- **Average Runtime per Exp**: {master_summary['runtime_statistics']['average_experiment_runtime_seconds']}s",
            "",
            "## Completed Experiments",
            "",
            "| ID | Name | Benchmark | Agent | Runs | Status | Runtime |",
            "|---|---|---|---|---|---|---|",
        ]

        for item in master_summary["completed_experiments"]:
            runs_str = f"{item.get('completed_runs', 0)}/{item.get('total_runs', 0)}"
            md_lines.append(
                f"| `{item['experiment_id'][:8]}` | {item['experiment_name']} | {item['benchmark']} | {item['agent']} | {runs_str} | {item['status']} | {item['runtime_seconds']}s |"
            )

        if master_summary["failed_experiments"]:
            md_lines.extend(
                [
                    "",
                    "## Failed Experiments",
                    "",
                    "| ID | Name | Benchmark | Agent | Error |",
                    "|---|---|---|---|---|",
                ]
            )
            for item in master_summary["failed_experiments"]:
                err_msg = item.get("error") or str(item.get("errors"))
                md_lines.append(
                    f"| `{item['experiment_id'][:8]}` | {item['experiment_name']} | {item['benchmark']} | {item['agent']} | {err_msg[:60]} |"
                )

        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        logger.info("Master summary written to %s and %s", json_path, md_path)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into HH:MM:SS format."""
        total_sec = int(round(seconds))
        hours = total_sec // 3600
        minutes = (total_sec % 3600) // 60
        secs = total_sec % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def _slugify(text: str) -> str:
        """Sanitize string for filenames."""
        return text.replace(" ", "_").replace("-", "_").replace(".", "_")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for running the Experiment Orchestrator."""
    parser = argparse.ArgumentParser(
        description="Declarative Experiment Orchestrator for LLM Reliability Ranking."
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML or JSON experiment definition file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="results",
        help="Root output directory (default: 'results').",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic resumption of interrupted experiments.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    orchestrator = ExperimentOrchestrator(output_dir=args.output_dir)
    result = orchestrator.run_from_file(args.config, resume=not args.no_resume)

    print("\nOrchestration Run Complete.")
    print(
        f"Total: {result['total_experiments']}, Completed: {result['completed_count']}, Failed: {result['failed_count']}"
    )
    print(f"Master Summary written to: {args.output_dir}")


if __name__ == "__main__":
    main()
