"""
Dataset verification for reproducibility.

Purpose
-------
Verify that dataset files exist and are intact by computing and comparing
SHA-256 checksums.  Supports both individual files and directories.

Usage example
-------------
>>> from llm_reliability.reproducibility.dataset_verifier import DatasetVerifier
>>> verifier = DatasetVerifier()
>>> result = verifier.verify(Path("datasets/my_data.json"))
>>> assert verifier.verify_checksum(Path("datasets/my_data.json"), result["sha256"])
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DatasetVerifier:
    """Verify dataset integrity via SHA-256 checksums.

    Parameters
    ----------
    buffer_size : int
        Buffer size (bytes) for streaming file hashing.
    """

    def __init__(self, buffer_size: int = 65536) -> None:
        self._buffer_size = buffer_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, path: Path) -> dict[str, Any]:
        """Check that *path* exists and compute its SHA-256 checksum.

        Parameters
        ----------
        path : Path
            File or directory to verify.

        Returns
        -------
        dict
            Keys: ``exists`` (bool), ``is_directory`` (bool), ``sha256`` (str
            or None for directories), ``path`` (str), ``error`` (str or None).
        """
        result: dict[str, Any] = {
            "path": str(path),
            "exists": False,
            "is_directory": False,
            "sha256": None,
            "error": None,
        }

        if not path.exists():
            result["error"] = f"Path does not exist: {path}"
            logger.warning(result["error"])
            return result

        result["exists"] = True

        if path.is_dir():
            result["is_directory"] = True
            result["sha256"] = self._hash_directory(path)
        elif path.is_file():
            result["sha256"] = self._hash_file(path)
        else:
            result["error"] = f"Path is neither a file nor a directory: {path}"

        return result

    def verify_checksum(self, path: Path, expected_sha256: str) -> bool:
        """Verify that the file at *path* matches *expected_sha256*.

        Parameters
        ----------
        path : Path
        expected_sha256 : str
            Expected hex-encoded SHA-256 digest.

        Returns
        -------
        bool
        """
        result = self.verify(path)

        if not result["exists"]:
            logger.error("Cannot verify checksum; path missing: %s", path)
            return False

        if result["sha256"] is None:
            logger.error("Cannot verify checksum for directory: %s", path)
            return False

        if result["sha256"] != expected_sha256:
            logger.error(
                "Checksum mismatch for %s: expected %s, got %s",
                path,
                expected_sha256,
                result["sha256"],
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_file(self, path: Path) -> str:
        """Compute SHA-256 digest of a file."""
        hasher = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(self._buffer_size)
                if not chunk:
                    break
                hasher.update(chunk)
        digest = hasher.hexdigest()
        logger.debug("SHA256 %s = %s", path, digest)
        return digest

    def _hash_directory(self, path: Path) -> str:
        """Compute a combined SHA-256 digest for all files in a directory.

        Sorts file paths for deterministic ordering, then hashes each file's
        relative path and content together.
        """
        hasher = hashlib.sha256()
        for child in sorted(path.rglob("*"), key=lambda p: str(p.relative_to(path))):
            if child.is_file():
                rel = str(child.relative_to(path))
                hasher.update(rel.encode("utf-8"))
                hasher.update(self._hash_file(child).encode("utf-8"))
        digest = hasher.hexdigest()
        logger.debug("SHA256 directory %s = %s", path, digest)
        return digest
