from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from llm_reliability.cache.backend import CacheBackend, FileSystemCacheBackend
from llm_reliability.configs.config import Configuration
from llm_reliability.logging.config import get_logger

if TYPE_CHECKING:
    from llm_reliability.pipeline.experiment_pipeline import ExperimentResult

logger = logging.getLogger(__name__)
log = get_logger(__name__)


class ExperimentCache:
    """Configurable cache for experiment results.

    Wraps a ``CacheBackend`` and provides key generation, automatic
    get-or-execute semantics, and runtime enable/disable control.

    Parameters
    ----------
    backend : CacheBackend
        The storage backend.  Defaults to ``FileSystemCacheBackend``.
    enabled : bool
        Whether caching is active.  When disabled, all operations are no-ops.
    """

    def __init__(
        self,
        backend: CacheBackend | None = None,
        enabled: bool = True,
    ) -> None:
        self._backend = backend or FileSystemCacheBackend()
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def backend(self) -> CacheBackend:
        return self._backend

    def generate_key(self, config: Configuration) -> str:
        """Return a deterministic cache key for the given configuration.

        The key is the SHA-256 hex digest of the configuration's canonical
        JSON representation, which already includes benchmark, agent, seed,
        perturbations, fault injection, and all other execution parameters.
        """
        return config.sha256()

    def get(self, key: str) -> ExperimentResult | None:
        """Retrieve a cached result by key, or ``None`` if missing."""
        if not self._enabled:
            return None
        return self._backend.get(key)

    def set(self, key: str, result: ExperimentResult) -> None:
        """Store a result under the given key."""
        if not self._enabled:
            return
        self._backend.set(key, result)

    def exists(self, key: str) -> bool:
        """Check whether a cache entry exists for the given key."""
        if not self._enabled:
            return False
        return self._backend.exists(key)

    def invalidate(self, key: str) -> None:
        """Remove a specific cache entry."""
        if not self._enabled:
            return
        self._backend.invalidate(key)

    def clear(self) -> None:
        """Remove all cache entries."""
        if not self._enabled:
            return
        self._backend.clear()

    def get_or_execute(
        self,
        config: Configuration,
        execute_fn: Callable[[], ExperimentResult],
    ) -> ExperimentResult:
        """Return cached result if available, otherwise run *execute_fn* and cache.

        Parameters
        ----------
        config : Configuration
            The experiment configuration used to derive the cache key.
        execute_fn : Callable[[], ExperimentResult]
            A callable that performs the actual experiment when cache misses.

        Returns
        -------
        ExperimentResult
        """
        if not self._enabled:
            return execute_fn()

        key = self.generate_key(config)
        cached = self.get(key)
        if cached is not None:
            logger.info("Cache HIT for key '%s' — returning cached result", key)
            log.info(
                "Cache hit",
                extra={
                    "event": "cache_hit",
                    "cache_key": key,
                    "benchmark": config.benchmark,
                    "agent": config.agent,
                    "seed": config.seed,
                },
            )
            return cached

        logger.info("Cache MISS for key '%s' — executing", key)
        log.info(
            "Cache miss",
            extra={
                "event": "cache_miss",
                "cache_key": key,
                "benchmark": config.benchmark,
                "agent": config.agent,
                "seed": config.seed,
            },
        )
        result = execute_fn()
        self.set(key, result)
        log.info(
            "Cache set",
            extra={
                "event": "cache_set",
                "cache_key": key,
                "benchmark": config.benchmark,
                "agent": config.agent,
                "seed": config.seed,
            },
        )
        return result
