"""
Scheduler for the Experiment Runner.

Generates an ordered sequence of (benchmark, agent, seed, run_index) run
descriptors from an ExperimentSpec, respecting the repetition count and
optional parallel execution flag.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_reliability.experiments.experiment_models import ExperimentSpec
from llm_reliability.experiments.seed_manager import SeedManager


@dataclass(frozen=True)
class RunDescriptor:
    """Immutable descriptor for a single experiment run."""

    benchmark_name: str
    agent_name: str
    base_seed: int
    run_index: int
    derived_seed: int
    dataset_path: str


class Scheduler:
    """Enumerates all runs described by an ExperimentSpec.

    The scheduling order is:
    1. benchmarks  (outer)
    2. agents      (middle)
    3. seeds       (inner-outer)
    4. repetitions (inner)

    This order is deterministic and platform-independent.
    """

    def __init__(self, spec: ExperimentSpec) -> None:
        self._spec = spec
        self._seed_manager = SeedManager(spec.seeds)

    def build_run_queue(self) -> list[RunDescriptor]:
        """Return the full ordered list of RunDescriptors.

        Returns
        -------
        list[RunDescriptor]
            Ordered list of run descriptors.
        """
        queue: list[RunDescriptor] = []
        for bspec in self._spec.benchmarks:
            for aspec in self._spec.agents:
                model = aspec.metadata.get("model") or aspec.agent_metadata.get("model")
                if model and ":" not in aspec.name:
                    agent_identifier = f"{aspec.name}:{model}"
                else:
                    agent_identifier = aspec.name
                for base_seed in self._spec.seeds:
                    for run_index in range(self._spec.repetitions):
                        derived = self._seed_manager.derive(
                            base_seed,
                            bspec.name,
                            agent_identifier,
                            run_index,
                        )
                        queue.append(
                            RunDescriptor(
                                benchmark_name=bspec.name,
                                agent_name=agent_identifier,
                                base_seed=base_seed,
                                run_index=run_index,
                                derived_seed=derived,
                                dataset_path=bspec.dataset_path,
                            )
                        )
        return queue

    def total_runs(self) -> int:
        """Return the total number of scheduled runs."""
        return (
            len(self._spec.benchmarks)
            * len(self._spec.agents)
            * len(self._spec.seeds)
            * self._spec.repetitions
        )
