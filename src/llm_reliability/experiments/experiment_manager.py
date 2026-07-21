"""
Experiment Manager.

Provides high-level lifecycle management for experiments: create, load,
save, archive, and list.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from llm_reliability.experiments.experiment_models import ExperimentSpec


class ExperimentManager:
    """Manages the lifecycle of experiment specifications on disk.

    Parameters
    ----------
    workspace : str | Path
        Root workspace directory where experiment specs are stored.
    """

    def __init__(self, workspace: str | Path = "experiments") -> None:
        self._workspace = Path(workspace)
        self._workspace.mkdir(parents=True, exist_ok=True)

    def create(self, spec: ExperimentSpec) -> ExperimentSpec:
        """Persist a new ExperimentSpec to the workspace.

        Parameters
        ----------
        spec : ExperimentSpec
            The experiment specification to create.

        Returns
        -------
        ExperimentSpec
            The persisted spec (unchanged).

        Raises
        ------
        FileExistsError
            If an experiment with the same ID already exists.
        """
        target = self._workspace / f"{spec.experiment_id}.json"
        if target.exists():
            raise FileExistsError(
                f"Experiment '{spec.experiment_id}' already exists at {target}."
            )
        target.write_text(spec.canonical_json(), encoding="utf-8")
        return spec

    def load(self, experiment_id: str) -> ExperimentSpec:
        """Load an ExperimentSpec by experiment ID.

        Parameters
        ----------
        experiment_id : str
            The unique experiment identifier.

        Returns
        -------
        ExperimentSpec

        Raises
        ------
        FileNotFoundError
            If the experiment does not exist.
        """
        target = self._workspace / f"{experiment_id}.json"
        if not target.exists():
            raise FileNotFoundError(f"No experiment found with id '{experiment_id}'.")
        return ExperimentSpec.from_canonical_json(target.read_text(encoding="utf-8"))

    def save(self, spec: ExperimentSpec) -> None:
        """Overwrite an existing ExperimentSpec on disk.

        Parameters
        ----------
        spec : ExperimentSpec
            The updated experiment specification.
        """
        target = self._workspace / f"{spec.experiment_id}.json"
        target.write_text(spec.canonical_json(), encoding="utf-8")

    def archive(self, experiment_id: str, archive_dir: str | Path = "archive") -> Path:
        """Move an experiment spec to an archive directory.

        Parameters
        ----------
        experiment_id : str
            The experiment ID to archive.
        archive_dir : str | Path
            Destination archive directory.

        Returns
        -------
        Path
            Path to the archived file.
        """
        src = self._workspace / f"{experiment_id}.json"
        if not src.exists():
            raise FileNotFoundError(f"Experiment '{experiment_id}' not found.")

        archive_path = Path(archive_dir)
        archive_path.mkdir(parents=True, exist_ok=True)
        dst = archive_path / src.name
        shutil.move(str(src), str(dst))
        return dst

    def list(self) -> list[str]:
        """Return IDs of all stored experiments.

        Returns
        -------
        list[str]
            Sorted list of experiment IDs.
        """
        return sorted(p.stem for p in self._workspace.glob("*.json"))
