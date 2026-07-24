"""Tests for ExperimentCache, CacheBackend, and pipeline integration."""

import logging
import tempfile
from pathlib import Path

import pytest

from llm_reliability.cache.backend import CacheBackend, FileSystemCacheBackend
from llm_reliability.cache.experiment_cache import ExperimentCache
from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
from llm_reliability.configs.config import Configuration
from llm_reliability.pipeline.experiment_pipeline import ExperimentPipeline, ExperimentResult
from tests.test_mock_benchmark import DummyAgent


logging.basicConfig(level=logging.DEBUG)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return Configuration(
        experiment_name="test_cache",
        benchmark="mock",
        agent="dummy",
        llm="test-llm",
        prompt_version="v1",
        dataset_version="v1",
        seed=42,
        repetitions=1,
    )


@pytest.fixture
def config_v2():
    """A different configuration to test distinct cache keys."""
    return Configuration(
        experiment_name="test_cache_v2",
        benchmark="mock",
        agent="dummy",
        llm="gpt-4",
        prompt_version="v2",
        dataset_version="v1",
        seed=99,
        repetitions=1,
    )


@pytest.fixture
def tmp_cache_dir():
    with tempfile.TemporaryDirectory(prefix="cache_test_") as tmp:
        yield Path(tmp)


@pytest.fixture
def backend(tmp_cache_dir):
    return FileSystemCacheBackend(cache_dir=tmp_cache_dir)


@pytest.fixture
def experiment_cache(backend):
    return ExperimentCache(backend=backend, enabled=True)


def _run_pipeline(config: Configuration, cache=None) -> ExperimentResult:
    benchmark = MockBenchmark(seed=config.seed)
    agent = DummyAgent()
    pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=agent, cache=cache)
    return pipeline.run()


# ---------------------------------------------------------------------------
# CacheBackend tests
# ---------------------------------------------------------------------------


class TestCacheBackend:
    def test_set_and_get(self, backend: CacheBackend, config: Configuration):
        key = config.sha256()
        result = _run_pipeline(config)
        backend.set(key, result)
        loaded = backend.get(key)
        assert loaded is not None
        assert loaded.configuration == config
        assert len(loaded.execution_records) == len(result.execution_records)

    def test_get_missing_returns_none(self, backend: CacheBackend):
        assert backend.get("nonexistent") is None

    def test_exists(self, backend: CacheBackend, config: Configuration):
        key = config.sha256()
        assert not backend.exists(key)
        result = _run_pipeline(config)
        backend.set(key, result)
        assert backend.exists(key)

    def test_invalidate(self, backend: CacheBackend, config: Configuration):
        key = config.sha256()
        result = _run_pipeline(config)
        backend.set(key, result)
        assert backend.exists(key)
        backend.invalidate(key)
        assert not backend.exists(key)

    def test_clear(self, backend: CacheBackend, config: Configuration, config_v2: Configuration):
        k1 = config.sha256()
        k2 = config_v2.sha256()
        backend.set(k1, _run_pipeline(config))
        backend.set(k2, _run_pipeline(config_v2))
        assert backend.exists(k1)
        assert backend.exists(k2)
        backend.clear()
        assert not backend.exists(k1)
        assert not backend.exists(k2)

    def test_custom_cache_dir(self, tmp_cache_dir: Path):
        custom_backend = FileSystemCacheBackend(cache_dir=tmp_cache_dir / "custom")
        assert (tmp_cache_dir / "custom").exists()
        key = "testkey"
        custom_backend.set(key, _run_pipeline(Configuration(
            experiment_name="custom_dir_test",
            benchmark="mock", agent="dummy", llm="t", prompt_version="1",
            dataset_version="1", seed=0, repetitions=1,
        )))
        assert custom_backend.exists(key)


# ---------------------------------------------------------------------------
# ExperimentCache tests
# ---------------------------------------------------------------------------


class TestExperimentCache:
    def test_generate_key_deterministic(self, experiment_cache: ExperimentCache, config: Configuration):
        k1 = experiment_cache.generate_key(config)
        k2 = experiment_cache.generate_key(config)
        assert k1 == k2

    def test_generate_key_differs(self, experiment_cache: ExperimentCache, config: Configuration, config_v2: Configuration):
        k1 = experiment_cache.generate_key(config)
        k2 = experiment_cache.generate_key(config_v2)
        assert k1 != k2

    def test_get_or_execute_on_miss(self, experiment_cache: ExperimentCache, config: Configuration):
        call_count = 0

        def execute():
            nonlocal call_count
            call_count += 1
            return _run_pipeline(config)

        result = experiment_cache.get_or_execute(config, execute)
        assert call_count == 1
        assert len(result.execution_records) > 0

    def test_get_or_execute_on_hit(self, experiment_cache: ExperimentCache, config: Configuration):
        call_count = 0

        def execute():
            nonlocal call_count
            call_count += 1
            return _run_pipeline(config)

        result1 = experiment_cache.get_or_execute(config, execute)
        assert call_count == 1

        result2 = experiment_cache.get_or_execute(config, execute)
        assert call_count == 1  # Should not call execute again
        assert result2.configuration == result1.configuration
        assert len(result2.execution_records) == len(result1.execution_records)

    def test_invalidate_forces_reexecution(self, experiment_cache: ExperimentCache, config: Configuration):
        call_count = 0

        def execute():
            nonlocal call_count
            call_count += 1
            return _run_pipeline(config)

        experiment_cache.get_or_execute(config, execute)
        assert call_count == 1

        key = experiment_cache.generate_key(config)
        experiment_cache.invalidate(key)

        experiment_cache.get_or_execute(config, execute)
        assert call_count == 2  # Re-executed after invalidation

    def test_clear_forces_reexecution(self, experiment_cache: ExperimentCache, config: Configuration):
        call_count = 0

        def execute():
            nonlocal call_count
            call_count += 1
            return _run_pipeline(config)

        experiment_cache.get_or_execute(config, execute)
        assert call_count == 1

        experiment_cache.clear()

        experiment_cache.get_or_execute(config, execute)
        assert call_count == 2

    def test_disabled_cache_no_ops(self, backend: CacheBackend, config: Configuration):
        cache = ExperimentCache(backend=backend, enabled=False)
        call_count = 0

        def execute():
            nonlocal call_count
            call_count += 1
            return _run_pipeline(config)

        result1 = cache.get_or_execute(config, execute)
        assert call_count == 1

        result2 = cache.get_or_execute(config, execute)
        assert call_count == 2  # Still executes because cache is disabled

        # get/set/exists/invalidate/clear should all be no-ops
        key = cache.generate_key(config)
        assert cache.get(key) is None
        assert not cache.exists(key)
        cache.set(key, result1)
        assert not cache.exists(key)
        cache.invalidate(key)
        cache.clear()  # Should not raise

    def test_enable_disable_toggle(self, backend: CacheBackend, config: Configuration):
        cache = ExperimentCache(backend=backend, enabled=True)
        call_count = 0

        def execute():
            nonlocal call_count
            call_count += 1
            return _run_pipeline(config)

        cache.get_or_execute(config, execute)
        assert call_count == 1

        cache.get_or_execute(config, execute)
        assert call_count == 1  # cached

        cache.enabled = False
        cache.get_or_execute(config, execute)
        assert call_count == 2  # not cached anymore

        cache.enabled = True
        cache.get_or_execute(config, execute)
        assert call_count == 2  # still in cache from before

    def test_deterministic_cache_key(self, experiment_cache: ExperimentCache, config: Configuration):
        key1 = experiment_cache.generate_key(config)
        key2 = experiment_cache.generate_key(config)
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 64  # SHA-256 hex digest

    def test_backward_compatible_no_cache(self, config: Configuration):
        """Pipeline works without any cache (backward compat)."""
        result = _run_pipeline(config)
        assert len(result.execution_records) > 0
        assert len(result.evaluation_records) > 0


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------


class TestPipelineCacheIntegration:
    def test_cache_hit_in_pipeline(self, experiment_cache: ExperimentCache, config: Configuration):
        benchmark = MockBenchmark(seed=config.seed)
        agent = DummyAgent()
        pipeline1 = ExperimentPipeline(config=config, benchmark=benchmark, agent=agent, cache=experiment_cache)
        result1 = pipeline1.run()
        assert len(result1.execution_records) > 0

        benchmark2 = MockBenchmark(seed=config.seed)
        agent2 = DummyAgent()
        pipeline2 = ExperimentPipeline(config=config, benchmark=benchmark2, agent=agent2, cache=experiment_cache)
        result2 = pipeline2.run()

        assert len(result2.execution_records) == len(result1.execution_records)
        for e1, e2 in zip(result1.execution_records, result2.execution_records):
            assert e1.sha256() == e2.sha256()

    def test_cache_miss_in_pipeline(self, experiment_cache: ExperimentCache, config: Configuration, config_v2: Configuration):
        """Different configs produce different cache entries."""
        r1 = _run_pipeline(config, cache=experiment_cache)
        r2 = _run_pipeline(config_v2, cache=experiment_cache)
        assert r1.configuration.experiment_name != r2.configuration.experiment_name

    def test_cache_invalidation_in_pipeline(self, experiment_cache: ExperimentCache, config: Configuration):
        _run_pipeline(config, cache=experiment_cache)
        key = experiment_cache.generate_key(config)
        assert experiment_cache.exists(key)

        experiment_cache.invalidate(key)
        assert not experiment_cache.exists(key)

        _run_pipeline(config, cache=experiment_cache)
        assert experiment_cache.exists(key)
