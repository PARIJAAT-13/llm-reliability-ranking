from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_reliability.pipeline.experiment_pipeline import ExperimentResult

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(".cache") / "experiment_cache"


class CacheBackend(ABC):
    """Abstract interface for experiment result cache storage."""

    @abstractmethod
    def get(self, key: str) -> ExperimentResult | None:
        ...

    @abstractmethod
    def set(self, key: str, result: ExperimentResult) -> None:
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        ...

    @abstractmethod
    def invalidate(self, key: str) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


class FileSystemCacheBackend(CacheBackend):
    """Filesystem-backed cache storing each entry as a JSON file.

    Each cache entry is stored as ``<cache_dir>/<key>.json``. The key is
    the SHA-256 hex digest of the experiment configuration.
    """

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return (self._cache_dir / key).with_suffix(".json")

    def get(self, key: str) -> ExperimentResult | None:
        from llm_reliability.pipeline.experiment_pipeline import ExperimentResult
        path = self._path(key)
        if not path.exists():
            return None
        try:
            data = path.read_text(encoding="utf-8")
            return ExperimentResult.from_canonical_json(data)
        except Exception as exc:
            logger.warning("Failed to load cached result for key '%s': %s", key, exc)
            return None

    def set(self, key: str, result: ExperimentResult) -> None:
        path = self._path(key)
        path.write_text(result.canonical_json(), encoding="utf-8")
        logger.debug("Cached experiment result with key '%s' (%d bytes)", key, path.stat().st_size)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def invalidate(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()
            logger.debug("Invalidated cache entry '%s'", key)

    def clear(self) -> None:
        count = 0
        for p in self._cache_dir.glob("*.json"):
            p.unlink()
            count += 1
        logger.debug("Cleared cache: %d entries removed", count)
