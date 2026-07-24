"""
Environment capture for reproducibility.

Purpose
-------
Capture a complete snapshot of the execution environment at the time an
experiment is run, including Python version, installed package versions,
operating system details, and hardware information.

Responsibilities
----------------
- Python interpreter version
- Installed package versions (via ``importlib.metadata``)
- OS name, version, and architecture
- CPU count and available memory
- Hostname (anonymised if requested)

Usage example
-------------
>>> from llm_reliability.reproducibility.environment import EnvironmentCapture
>>> env = EnvironmentCapture.capture()
>>> env_dict = env.to_dict()

How environment is captured
---------------------------
``EnvironmentCapture.capture()`` is a class method that interrogates the
running Python interpreter via ``sys``, ``platform``, ``importlib.metadata``,
and ``os``.  The result is a Pydantic model that can be serialised to JSON.
"""

from __future__ import annotations

import logging
import os
import platform
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EnvironmentCapture(BaseModel):
    """Snapshot of the execution environment.

    Attributes
    ----------
    captured_at : str
        ISO-8601 UTC timestamp.
    python_version : str
        Full Python version string, e.g. ``"3.11.5"``.
    python_implementation : str
        Interpreter implementation, e.g. ``"CPython"``.
    platform_system : str
        OS name, e.g. ``"Linux"``, ``"Windows"``.
    platform_release : str
        OS release, e.g. ``"22.04"``.
    platform_machine : str
        CPU architecture, e.g. ``"x86_64"``.
    cpu_count : int | None
        Number of logical CPU cores.
    packages : dict[str, str]
        Installed package name → version mapping.
    git_commit : str | None
        Current HEAD git commit hash, if available.
    """

    model_config = {"arbitrary_types_allowed": True}

    captured_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    python_version: str
    python_implementation: str
    platform_system: str
    platform_release: str
    platform_machine: str
    cpu_count: int | None = None
    packages: dict[str, str] = Field(default_factory=dict)
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return self.model_dump()

    @classmethod
    def capture(cls, include_all_packages: bool = False) -> EnvironmentCapture:
        """Capture the current execution environment.

        Parameters
        ----------
        include_all_packages : bool
            If True, include all installed packages.  If False, only include
            packages relevant to the framework (pydantic, numpy, pandas, etc.).

        Returns
        -------
        EnvironmentCapture
        """
        packages = cls._capture_packages(include_all_packages)
        git_commit = cls._capture_git_commit()

        return cls(
            python_version=platform.python_version(),
            python_implementation=platform.python_implementation(),
            platform_system=platform.system(),
            platform_release=platform.release(),
            platform_machine=platform.machine(),
            cpu_count=os.cpu_count(),
            packages=packages,
            git_commit=git_commit,
        )

    @staticmethod
    def _capture_packages(include_all: bool) -> dict[str, str]:
        """Return a mapping of package name → version."""
        _relevant = {
            "pydantic",
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "scipy",
            "statsmodels",
            "plotly",
            "openpyxl",
            "tabulate",
            "pytest",
            "llm-reliability-ranking",
        }
        packages: dict[str, str] = {}
        try:
            import importlib.metadata as meta

            for dist in meta.distributions():
                name = dist.metadata.get("Name", "")
                version = dist.metadata.get("Version", "")
                if name and version:
                    if include_all or name.lower() in _relevant:
                        packages[name.lower()] = version
        except Exception:
            pass
        return packages

    @staticmethod
    def _capture_git_commit() -> str | None:
        """Return the current HEAD commit hash, or None if git is unavailable."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as exc:
            logger.debug("Could not capture git commit: %s", exc)
        return None
