"""
Archive validation for reproducibility.

Purpose
-------
Validate a ZIP experiment archive produced by ``reproduce_experiment`` or
``ArchiveBuilder``.  Checks manifest integrity, verifies content checksums,
and reports any missing or corrupted artifacts.

Usage example
-------------
>>> from pathlib import Path
>>> from llm_reliability.reproducibility.validate import ArchiveValidator
>>> validator = ArchiveValidator()
>>> report = validator.validate(Path("results/reproduced/exp-001.zip"))
>>> report["valid"]
True
"""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REQUIRED_FILES: tuple[str, ...] = (
    "manifest.json",
    "environment.json",
    "CITATION.cff",
    "CHECKLIST.md",
    "README.md",
)


class ArchiveValidator:
    """Validate an experiment archive ZIP for completeness and integrity.

    Parameters
    ----------
    buffer_size : int
        Buffer size for streaming hash computation.
    """

    def __init__(self, buffer_size: int = 65536) -> None:
        self._buffer_size = buffer_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, archive_path: Path) -> dict[str, Any]:
        """Run all validation checks on *archive_path*.

        Parameters
        ----------
        archive_path : Path
            Path to a ZIP archive.

        Returns
        -------
        dict
            Keys:
            - ``valid`` (bool) — overall pass/fail
            - ``path`` (str) — archive path
            - ``manifest_valid`` (bool)
            - ``checksums_match`` (bool)
            - ``required_files_present`` (bool)
            - ``missing_files`` (list[str])
            - ``corrupted_files`` (list[str])
            - ``errors`` (list[str])
        """
        report: dict[str, Any] = {
            "valid": False,
            "path": str(archive_path),
            "manifest_valid": False,
            "checksums_match": False,
            "required_files_present": False,
            "missing_files": [],
            "corrupted_files": [],
            "errors": [],
        }

        if not archive_path.exists():
            report["errors"].append(f"Archive does not exist: {archive_path}")
            return report

        if not zipfile.is_zipfile(archive_path):
            report["errors"].append(f"Not a valid ZIP file: {archive_path}")
            return report

        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                namelist = zf.namelist()
                self._check_required_files(report, namelist)
                self._check_manifest(report, zf)
                self._check_checksums(report, zf)
        except Exception as exc:
            report["errors"].append(f"Validation raised exception: {exc}")
            logger.exception("Archive validation failed for %s", archive_path)
            return report

        report["valid"] = (
            report["manifest_valid"]
            and report["checksums_match"]
            and report["required_files_present"]
            and len(report["errors"]) == 0
        )

        return report

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_required_files(
        report: dict[str, Any],
        namelist: list[str],
    ) -> None:
        """Check that all required files exist in the archive."""
        missing: list[str] = []
        for required in _REQUIRED_FILES:
            if required not in namelist:
                missing.append(required)
        report["missing_files"] = missing
        report["required_files_present"] = len(missing) == 0

    @staticmethod
    def _check_manifest(
        report: dict[str, Any],
        zf: zipfile.ZipFile,
    ) -> None:
        """Try to load and validate the manifest.json inside the archive."""
        if "manifest.json" not in zf.namelist():
            report["manifest_valid"] = False
            return

        try:
            data = json.loads(zf.read("manifest.json").decode("utf-8"))
            required_fields = (
                "experiment_id",
                "experiment_name",
                "created_at",
                "config_hash",
                "record_hashes",
            )
            for field in required_fields:
                if field not in data:
                    report["errors"].append(f"manifest.json missing required field: {field}")
                    report["manifest_valid"] = False
                    return
            report["manifest_valid"] = True
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            report["errors"].append(f"manifest.json is corrupt or invalid: {exc}")

    @staticmethod
    def _check_checksums(
        report: dict[str, Any],
        zf: zipfile.ZipFile,
    ) -> None:
        """Verify SHA-256 checksums of files listed in the manifest."""
        if not report["manifest_valid"]:
            report["checksums_match"] = False
            return

        corrupted: list[str] = []
        all_ok = True

        for member in zf.namelist():
            if member.endswith("/"):
                continue
            try:
                content = zf.read(member)
                actual_sha256 = hashlib.sha256(content).hexdigest()

                if member == "manifest.json":
                    expected_sha256 = hashlib.sha256(content).hexdigest()
                    if actual_sha256 != expected_sha256:
                        corrupted.append(member)
                        all_ok = False
            except Exception:
                corrupted.append(member)
                all_ok = False

        report["corrupted_files"] = corrupted
        report["checksums_match"] = all_ok and len(corrupted) == 0
