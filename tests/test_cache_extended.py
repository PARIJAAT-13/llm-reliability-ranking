"""Extended tests for FileSystemCacheBackend and ExperimentCache edge cases."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from llm_reliability.cache import ExperimentCache, FileSystemCacheBackend
from llm_reliability.configs import Configuration
from llm_reliability.pipeline.experiment_pipeline import ExperimentResult
from tests.conftest import make_configuration

pytestmark = pytest.mark.usefixtures("_ensure_mock_benchmark")


@pytest.fixture(scope="session", autouse=True)
def _ensure_mock_benchmark():
    from llm_reliability.benchmarks import mock_benchmark  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> Configuration:
    return make_configuration()


@pytest.fixture
def config_v2() -> Configuration:
    return make_configuration(
        experiment_name="pilot_v2",
        seed=99,
        llm="gpt-4",
        prompt_version="v2",
    )


@pytest.fixture
def result(config: Configuration) -> ExperimentResult:
    return ExperimentResult(
        configuration=config,
        execution_records=[],
        evaluation_records=[],
        metric_records=[],
        ranking_records=[],
    )


@pytest.fixture
def result_v2(config_v2: Configuration) -> ExperimentResult:
    return ExperimentResult(
        configuration=config_v2,
        execution_records=[],
        evaluation_records=[],
        metric_records=[],
        ranking_records=[],
    )


@pytest.fixture
def backend(tmp_path: Path) -> FileSystemCacheBackend:
    return FileSystemCacheBackend(cache_dir=tmp_path)


@pytest.fixture
def experiment_cache(backend: FileSystemCacheBackend) -> ExperimentCache:
    return ExperimentCache(backend=backend, enabled=True)


# ---------------------------------------------------------------------------
# Direct set/get round-trip
# ---------------------------------------------------------------------------


class TestCacheSetGet:
    def test_cache_set_get_roundtrip(
        self, backend: FileSystemCacheBackend, result: ExperimentResult, config: Configuration
    ):
        key = config.sha256()
        backend.set(key, result)
        loaded = backend.get(key)
        assert loaded is not None
        assert loaded.configuration == config

    def test_cache_miss(self, backend: FileSystemCacheBackend):
        assert backend.get("nonexistent-key") is None

    def test_cache_overwrite(
        self,
        backend: FileSystemCacheBackend,
        result: ExperimentResult,
        result_v2: ExperimentResult,
        config_v2: Configuration,
    ):
        key = config_v2.sha256()
        backend.set(key, result)
        backend.set(key, result_v2)
        loaded = backend.get(key)
        assert loaded is not None
        assert loaded.configuration.experiment_name == "pilot_v2"


# ---------------------------------------------------------------------------
# Invalidation and clearing
# ---------------------------------------------------------------------------


class TestCacheInvalidateClear:
    def test_cache_invalidate(
        self, backend: FileSystemCacheBackend, result: ExperimentResult, config: Configuration
    ):
        key = config.sha256()
        backend.set(key, result)
        assert backend.exists(key)
        backend.invalidate(key)
        assert not backend.exists(key)
        assert backend.get(key) is None

    def test_cache_clear(
        self,
        backend: FileSystemCacheBackend,
        result: ExperimentResult,
        result_v2: ExperimentResult,
        config: Configuration,
        config_v2: Configuration,
    ):
        k1 = config.sha256()
        k2 = config_v2.sha256()
        backend.set(k1, result)
        backend.set(k2, result_v2)
        backend.clear()
        assert not backend.exists(k1)
        assert not backend.exists(k2)
        assert backend.get(k1) is None
        assert backend.get(k2) is None

    def test_cache_invalidate_nonexistent_key(self, backend: FileSystemCacheBackend):
        backend.invalidate("i-do-not-exist")


# ---------------------------------------------------------------------------
# Disabled cache
# ---------------------------------------------------------------------------


class TestCacheDisabled:
    def test_cache_disabled(
        self, backend: FileSystemCacheBackend, result: ExperimentResult, config: Configuration
    ):
        cache = ExperimentCache(backend=backend, enabled=False)
        key = cache.generate_key(config)
        assert cache.get(key) is None
        assert not cache.exists(key)
        cache.set(key, result)
        assert not cache.exists(key)
        cache.invalidate(key)
        cache.clear()


# ---------------------------------------------------------------------------
# Key uniqueness
# ---------------------------------------------------------------------------


class TestCacheKeyUniqueness:
    def test_cache_key_uniqueness(
        self,
        experiment_cache: ExperimentCache,
        config: Configuration,
        config_v2: Configuration,
    ):
        k1 = experiment_cache.generate_key(config)
        k2 = experiment_cache.generate_key(config_v2)
        assert k1 != k2
        assert len(k1) == 64
        assert len(k2) == 64


# ---------------------------------------------------------------------------
# Serialisation / deserialisation failures
# ---------------------------------------------------------------------------


class TestCacheSerialization:
    def test_cache_serialization_failure(
        self, backend: FileSystemCacheBackend, result: ExperimentResult, config: Configuration
    ):
        key = config.sha256()
        with patch.object(result.__class__, "canonical_json", side_effect=ValueError("boom")):
            with pytest.raises(ValueError, match="boom"):
                backend.set(key, result)

    def test_cache_deserialization_failure(
        self, backend: FileSystemCacheBackend, result: ExperimentResult, config: Configuration
    ):
        key = config.sha256()
        backend.set(key, result)
        path = backend._path(key)
        path.write_text("{corrupt: json}", encoding="utf-8")
        loaded = backend.get(key)
        assert loaded is None


# ---------------------------------------------------------------------------
# Persistence across instances
# ---------------------------------------------------------------------------


class TestCachePersistence:
    def test_cache_persistence(
        self, tmp_path: Path, result: ExperimentResult, config: Configuration
    ):
        key = config.sha256()
        backend1 = FileSystemCacheBackend(cache_dir=tmp_path)
        backend1.set(key, result)

        backend2 = FileSystemCacheBackend(cache_dir=tmp_path)
        loaded = backend2.get(key)
        assert loaded is not None
        assert loaded.configuration == config


# ---------------------------------------------------------------------------
# Path-traversal prevention
# ---------------------------------------------------------------------------


class TestBackendPathTraversal:
    def test_backend_path_traversal_escapes_cache_dir(self, tmp_path: Path):
        backend = FileSystemCacheBackend(cache_dir=tmp_path)
        malicious_key = "../../etc/passwd"
        path = backend._path(malicious_key)
        resolved = path.resolve()
        safe_dir = tmp_path.resolve()
        assert safe_dir not in resolved.parents
