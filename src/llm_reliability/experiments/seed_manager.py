"""
Seed Manager for deterministic experiment execution.

Generates reproducible seeds for each (benchmark, agent, run_index) triple
using a deterministic derivation from the base seed, ensuring that repeated
runs are bitwise-identical regardless of execution order.
"""

from __future__ import annotations

import hashlib


class SeedManager:
    """Derives deterministic seeds for each experiment run triple.

    Algorithm
    ---------
    For each combination of (base_seed, benchmark_name, agent_name, run_index),
    a unique 32-bit seed is derived by hashing the canonical string
    ``"{base_seed}:{benchmark}:{agent}:{run_index}"`` with SHA-256 and
    taking the first 8 hex characters as an unsigned integer.

    This guarantees:
    - Same output for identical inputs on any platform.
    - Statistically independent seeds across triples.
    - No seed reuse for different (benchmark, agent, run_index) combinations.
    """

    def __init__(self, base_seeds: list[int]) -> None:
        """Initialize with one or more base seeds.

        Parameters
        ----------
        base_seeds : list[int]
            Non-empty list of non-negative integers.
        """
        if not base_seeds:
            raise ValueError("SeedManager requires at least one base seed.")
        for s in base_seeds:
            if s < 0:
                raise ValueError(f"Seeds must be non-negative, got {s}.")
        self._base_seeds = list(base_seeds)

    @property
    def base_seeds(self) -> list[int]:
        """Return the list of base seeds."""
        return list(self._base_seeds)

    def derive(
        self,
        base_seed: int,
        benchmark: str,
        agent: str,
        run_index: int,
    ) -> int:
        """Derive a deterministic seed for a specific (benchmark, agent, run).

        Parameters
        ----------
        base_seed : int
            The primary seed from ExperimentSpec.seeds.
        benchmark : str
            The benchmark name.
        agent : str
            The agent name.
        run_index : int
            Zero-based repetition index.

        Returns
        -------
        int
            A non-negative 32-bit seed value.
        """
        key = f"{base_seed}:{benchmark}:{agent}:{run_index}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:8]
        return int(digest, 16)

    def all_seeds_for(
        self,
        benchmark: str,
        agent: str,
        repetitions: int,
    ) -> list[int]:
        """Return all derived seeds across all base seeds for (benchmark, agent).

        Parameters
        ----------
        benchmark : str
            The benchmark name.
        agent : str
            The agent name.
        repetitions : int
            Number of repetitions per base seed.

        Returns
        -------
        list[int]
            Flat list of derived seeds in (base_seed, run_index) order.
        """
        seeds = []
        for base_seed in self._base_seeds:
            for run_index in range(repetitions):
                seeds.append(self.derive(base_seed, benchmark, agent, run_index))
        return seeds
