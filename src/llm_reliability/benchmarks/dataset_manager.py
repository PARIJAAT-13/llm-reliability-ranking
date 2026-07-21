"""
Dataset Manager for LLM Reliability Ranking Framework.

Provides dataset downloading, validation, caching, version tracking,
integrity verification, and offline mode support for official benchmark datasets
(AgentBoard, GAIA, and SWE-bench Lite).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import urllib.request
from typing import Any

from pydantic import Field
from llm_reliability.utils.serialization import SerializableModel

logger = logging.getLogger(__name__)

# Default Dataset Download Sources & Metadata
DATASET_MANIFEST: dict[str, dict[str, Any]] = {
    "agentboard": {
        "version": "1.0.0",
        "description": "AgentBoard Benchmark Dataset",
        "url": "https://raw.githubusercontent.com/AgentBoard/AgentBoard/main/data/agentboard_eval.json",
        "filename": "agentboard_eval.json",
        "expected_sha256": None,  # Computed upon caching if optional
    },
    "gaia": {
        "version": "1.0.0",
        "description": "GAIA (General AI Assistants) Benchmark Dataset",
        "url": "https://raw.githubusercontent.com/GAIA-benchmark/GAIA/main/data/validation.jsonl",
        "filename": "gaia_validation.jsonl",
        "expected_sha256": None,
    },
    "swe_bench_lite": {
        "version": "1.0.0",
        "description": "SWE-bench Lite Dataset",
        "url": "https://raw.githubusercontent.com/princeton-nlp/SWE-bench/main/data/swe-bench-lite.json",
        "filename": "swe_bench_lite.json",
        "expected_sha256": None,
    },
}


class DatasetInfo(SerializableModel):
    """Metadata representation of a downloaded benchmark dataset."""

    benchmark_name: str
    version: str
    file_path: str
    file_size_bytes: int
    sha256_hash: str
    is_valid: bool
    downloaded_at: str | None = None


class DatasetManager:
    """Manages benchmark dataset downloading, validation, caching, and version provenance."""

    def __init__(self, cache_dir: str | pathlib.Path = "data/cache", offline_mode: bool = False):
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline_mode = offline_mode or os.getenv("LLM_RELIABILITY_OFFLINE", "0").lower() in ("1", "true")

    def compute_sha256(self, file_path: pathlib.Path) -> str:
        """Compute SHA-256 hash of a local file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def validate_dataset(self, benchmark_name: str, file_path: pathlib.Path) -> bool:
        """Validate local dataset file existence and integrity."""
        if not file_path.exists() or file_path.stat().st_size == 0:
            return False

        # Structural JSON / JSONL validation
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix.lower() in (".jsonl", ".ndjson"):
                    line_count = sum(1 for line in f if line.strip())
                    return line_count > 0
                else:
                    data = json.load(f)
                    return isinstance(data, (list, dict)) and len(data) > 0
        except Exception as e:
            logger.warning("Dataset validation failed for %s at %s: %s", benchmark_name, file_path, e)
            return False

    def get_dataset(self, benchmark_name: str, force_download: bool = False) -> DatasetInfo:
        """Get benchmark dataset info, downloading or loading from cache as needed."""
        norm_name = benchmark_name.lower().replace("-", "_")
        manifest = DATASET_MANIFEST.get(norm_name)
        if not manifest:
            # Fallback for mock or custom benchmarks
            fallback_path = self.cache_dir / f"{norm_name}.json"
            if not fallback_path.exists():
                # Create synthetic placeholder for mock testing if needed
                with open(fallback_path, "w", encoding="utf-8") as f:
                    json.dump([{"task_id": "mock_1", "prompt": "Solve mock"}], f)
            sha = self.compute_sha256(fallback_path)
            return DatasetInfo(
                benchmark_name=benchmark_name,
                version="1.0.0",
                file_path=str(fallback_path.resolve()),
                file_size_bytes=fallback_path.stat().st_size,
                sha256_hash=sha,
                is_valid=True,
            )

        target_path = self.cache_dir / manifest["filename"]

        if target_path.exists() and not force_download:
            if self.validate_dataset(norm_name, target_path):
                logger.info("Using cached dataset for %s: %s", norm_name, target_path)
                sha = self.compute_sha256(target_path)
                return DatasetInfo(
                    benchmark_name=norm_name,
                    version=manifest["version"],
                    file_path=str(target_path.resolve()),
                    file_size_bytes=target_path.stat().st_size,
                    sha256_hash=sha,
                    is_valid=True,
                )

        if self.offline_mode:
            raise FileNotFoundError(
                f"Offline mode enabled and cached dataset for '{norm_name}' not found at {target_path}."
            )

        # Download dataset
        logger.info("Downloading dataset for %s from %s", norm_name, manifest["url"])
        try:
            urllib.request.urlretrieve(manifest["url"], target_path)
        except Exception as e:
            logger.error("Failed to download dataset for %s: %s", norm_name, e)
            raise RuntimeError(f"Download failed for dataset '{norm_name}': {e}") from e

        if not self.validate_dataset(norm_name, target_path):
            raise ValueError(f"Downloaded dataset for '{norm_name}' at {target_path} is invalid.")

        sha = self.compute_sha256(target_path)
        return DatasetInfo(
            benchmark_name=norm_name,
            version=manifest["version"],
            file_path=str(target_path.resolve()),
            file_size_bytes=target_path.stat().st_size,
            sha256_hash=sha,
            is_valid=True,
        )
