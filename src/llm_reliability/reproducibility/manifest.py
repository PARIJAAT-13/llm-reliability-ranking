"""
Experiment manifest generator.

Purpose
-------
Produce a ``manifest.json`` file that captures every artifact produced by
an experiment run so that any researcher can verify and reproduce the results.

Responsibilities
----------------
- Experiment ID, name, and timestamps
- Git commit hash and Python version
- Seeds used in the experiment
- SHA-256 content hashes of all records (executions, evaluations, metrics, rankings)
- Configuration snapshot hash
- Environment snapshot reference

Usage example
-------------
>>> from llm_reliability.reproducibility.manifest import ManifestGenerator
>>> gen = ManifestGenerator()
>>> manifest = gen.build(summary, environment)
>>> gen.save(manifest, "results/exp-001/manifest.json")

How manifests are produced
--------------------------
``build()`` calls ``sha256()`` on every ``SerializableModel`` record and
aggregates them into a ``Manifest`` Pydantic model.  Records that do not
expose ``sha256()`` are identified by their string representation.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class RecordHashes(BaseModel):
    """SHA-256 hashes of all experiment artifacts."""

    executions: list[str] = Field(default_factory=list)
    evaluations: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    rankings: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    """Complete experiment reproducibility manifest.

    Attributes
    ----------
    manifest_version : str
        Schema version of this manifest.
    experiment_id : str
        Unique experiment identifier.
    experiment_name : str
        Human-readable name.
    created_at : str
        ISO-8601 UTC creation timestamp.
    git_commit : str | None
        HEAD commit hash at run time.
    python_version : str
        Python interpreter version.
    seeds : list[int]
        All seeds used in the experiment.
    config_hash : str
        SHA-256 of the configuration snapshot.
    record_hashes : RecordHashes
        Hashes of all serialised records.
    environment_file : str
        Relative path to the ``environment.json`` file.
    """

    manifest_version: str = "1.0"
    experiment_id: str
    experiment_name: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    python_version: str = ""
    seeds: list[int] = Field(default_factory=list)
    config_hash: str = ""
    record_hashes: RecordHashes = Field(default_factory=RecordHashes)
    environment_file: str = "environment.json"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ManifestGenerator:
    """Builds and saves experiment manifests."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        summary: Any,
        environment: Any | None = None,
    ) -> Manifest:
        """Build a ``Manifest`` from an ``ExperimentSummary``.

        Parameters
        ----------
        summary : ExperimentSummary
            Aggregated experiment data.
        environment : EnvironmentCapture | None
            Captured environment snapshot.

        Returns
        -------
        Manifest
        """
        config_hash = self._hash_dict(summary.config_snapshot)
        record_hashes = RecordHashes(
            executions=self._hash_records(summary.executions),
            evaluations=self._hash_records(summary.evaluations),
            metrics=self._hash_records(summary.metrics),
            rankings=self._hash_records(summary.rankings),
        )

        seeds = self._extract_seeds(summary)
        git_commit = None
        python_version = ""
        if environment is not None:
            git_commit = getattr(environment, "git_commit", None)
            python_version = getattr(environment, "python_version", "")

        return Manifest(
            experiment_id=summary.experiment_id,
            experiment_name=summary.experiment_name,
            git_commit=git_commit,
            python_version=python_version,
            seeds=seeds,
            config_hash=config_hash,
            record_hashes=record_hashes,
            metadata=dict(summary.metadata),
        )

    def save(
        self,
        manifest: Manifest,
        path: str | pathlib.Path,
        indent: int = 2,
    ) -> pathlib.Path:
        """Serialise the manifest to ``manifest.json``.

        Parameters
        ----------
        manifest : Manifest
        path : str | Path
            Output file path.
        indent : int
            JSON indentation.

        Returns
        -------
        pathlib.Path
        """
        dest = pathlib.Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = manifest.model_dump()
        dest.write_text(json.dumps(data, indent=indent, default=str), encoding="utf-8")
        return dest

    def load(self, path: str | pathlib.Path) -> Manifest:
        """Load a manifest from disk.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        Manifest
        """
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return Manifest.model_validate(data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_records(records: list[Any]) -> list[str]:
        """Return a list of SHA-256 hashes, one per record."""
        hashes = []
        for rec in records:
            if hasattr(rec, "sha256"):
                hashes.append(rec.sha256())
            else:
                # Fallback: hash the string repr
                hashes.append(hashlib.sha256(str(rec).encode("utf-8")).hexdigest())
        return hashes

    @staticmethod
    def _hash_dict(data: dict[str, Any]) -> str:
        """SHA-256 hash of a dictionary (JSON-serialised, sorted keys)."""
        raw = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _extract_seeds(summary: Any) -> list[int]:
        """Attempt to extract seeds from evaluations or config."""
        seeds_seen: set[int] = set()
        for ev in getattr(summary, "evaluations", []):
            seed = getattr(ev, "seed", None)
            if isinstance(seed, int):
                seeds_seen.add(seed)
        return sorted(seeds_seen)
