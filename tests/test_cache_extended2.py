"""Extended tests for ExperimentCache and FileSystemCacheBackend — edge cases and error handling."""

from __future__ import annotations

import json
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


class TestCacheHitMiss:
    def test_cache_hit(
        self, experiment_cache: ExperimentCache, config: Configuration, result: ExperimentResult
    ):
        key = experiment_cache.generate_key(config)
        experiment_cache.set(key, result)
        cached = experiment_cache.get(key)
        assert cached is not None
        assert cached.configuration == config

    def test_cache_miss(self, experiment_cache: ExperimentCache):
        assert experiment_cache.get("nonexistent") is None

    def test_cache_miss_when_disabled(
        self, backend: FileSystemCacheBackend, config: Configuration, result: ExperimentResult
    ):
        cache = ExperimentCache(backend=backend, enabled=False)
        key = cache.generate_key(config)
        cache.set(key, result)
        assert cache.get(key) is None

    def test_cache_exists_returns_true_for_present_key(
        self, experiment_cache: ExperimentCache, config: Configuration, result: ExperimentResult
    ):
        key = experiment_cache.generate_key(config)
        experiment_cache.set(key, result)
        assert experiment_cache.exists(key) is True

    def test_cache_exists_returns_false_for_absent_key(self, experiment_cache: ExperimentCache):
        assert experiment_cache.exists("no-such-key") is False


class TestCacheKeyGeneration:
    def test_same_config_same_key(self, experiment_cache: ExperimentCache, config: Configuration):
        k1 = experiment_cache.generate_key(config)
        k2 = experiment_cache.generate_key(config)
        assert k1 == k2

    def test_different_configs_different_keys(
        self, experiment_cache: ExperimentCache, config: Configuration, config_v2: Configuration
    ):
        k1 = experiment_cache.generate_key(config)
        k2 = experiment_cache.generate_key(config_v2)
        assert k1 != k2

    def test_key_is_hex_string(self, experiment_cache: ExperimentCache, config: Configuration):
        key = experiment_cache.generate_key(config)
        assert all(c in "0123456789abcdef" for c in key)
        assert len(key) > 0

    def test_key_deterministic_across_instances(self, config: Configuration):
        cache1 = ExperimentCache()
        cache2 = ExperimentCache()
        assert cache1.generate_key(config) == cache2.generate_key(config)


class TestGetOrExecute:
    def test_get_or_execute_cache_hit(
        self, experiment_cache: ExperimentCache, config: Configuration, result: ExperimentResult
    ):
        key = experiment_cache.generate_key(config)
        experiment_cache.set(key, result)
        call_count = 0

        def execute_fn():
            nonlocal call_count
            call_count += 1
            return result

        cached_result = experiment_cache.get_or_execute(config, execute_fn)
        assert cached_result.configuration == config
        assert call_count == 0

    def test_get_or_execute_cache_miss(
        self, experiment_cache: ExperimentCache, config: Configuration, result: ExperimentResult
    ):
        call_count = 0

        def execute_fn():
            nonlocal call_count
            call_count += 1
            return result

        cached_result = experiment_cache.get_or_execute(config, execute_fn)
        assert cached_result.configuration == config
        assert call_count == 1

    def test_get_or_execute_skips_cache_when_disabled(
        self, backend: FileSystemCacheBackend, config: Configuration, result: ExperimentResult
    ):
        cache = ExperimentCache(backend=backend, enabled=False)
        call_count = 0

        def execute_fn():
            nonlocal call_count
            call_count += 1
            return result

        cache.get_or_execute(config, execute_fn)
        assert call_count == 1

    def test_get_or_execute_side_effects(
        self, experiment_cache: ExperimentCache, config: Configuration, result: ExperimentResult
    ):
        side_effects = []

        def execute_fn():
            side_effects.append("executed")
            return result

        experiment_cache.get_or_execute(config, execute_fn)
        assert side_effects == ["executed"]

        experiment_cache.get_or_execute(config, execute_fn)
        assert side_effects == ["executed"]


class TestCacheEnableDisable:
    def test_enabled_property_defaults_to_true(self):
        cache = ExperimentCache()
        assert cache.enabled is True

    def test_disable_via_setter(self, backend: FileSystemCacheBackend):
        cache = ExperimentCache(backend=backend, enabled=True)
        assert cache.enabled is True
        cache.enabled = False
        assert cache.enabled is False
        cache.enabled = True
        assert cache.enabled is True

    def test_disabled_cache_does_not_store(
        self, backend: FileSystemCacheBackend, config: Configuration, result: ExperimentResult
    ):
        cache = ExperimentCache(backend=backend, enabled=False)
        key = cache.generate_key(config)
        cache.set(key, result)
        assert not cache.exists(key)

    def test_disabled_cache_clear_is_noop(self, backend: FileSystemCacheBackend):
        cache = ExperimentCache(backend=backend, enabled=False)
        cache.clear()

    def test_disabled_cache_invalidate_is_noop(self, backend: FileSystemCacheBackend):
        cache = ExperimentCache(backend=backend, enabled=False)
        cache.invalidate("some-key")


class TestCacheDirectoryCreation:
    def test_backend_creates_cache_dir(self, tmp_path: Path):
        cache_dir = tmp_path / "my_cache"
        assert not cache_dir.exists()
        FileSystemCacheBackend(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_backend_uses_existing_cache_dir(self, tmp_path: Path):
        cache_dir = tmp_path / "existing_cache"
        cache_dir.mkdir(parents=True)
        FileSystemCacheBackend(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_default_cache_dir_is_dot_cache(self):
        backend = FileSystemCacheBackend()
        assert ".cache" in str(backend._cache_dir)


class TestCacheSerializationRoundTrip:
    def test_serialization_round_trip(self, backend: FileSystemCacheBackend, config: Configuration):
        original = ExperimentResult(
            configuration=config,
            execution_records=[],
            evaluation_records=[],
            metric_records=[],
            ranking_records=[],
        )
        key = config.sha256()
        backend.set(key, original)
        loaded = backend.get(key)
        assert loaded is not None
        assert loaded.configuration.experiment_name == original.configuration.experiment_name
        assert loaded.configuration.seed == original.configuration.seed

    def test_serialization_with_records(
        self, backend: FileSystemCacheBackend, config: Configuration
    ):
        from llm_reliability.records.evaluation import EvaluationRecord
        from llm_reliability.records.execution import ExecutionRecord
        from llm_reliability.records.metric import MetricRecord
        from llm_reliability.records.ranking import RankingRecord

        exec_rec = ExecutionRecord(
            configuration_hash="a" * 64,
            seed=42,
            benchmark="mock",
            agent="test",
            task_id="t1",
            run_index=0,
            runtime_seconds=1.0,
            timestamp="2026-01-01T00:00:00+00:00",
            status="success",
            agent_output="output",
        )
        eval_rec = EvaluationRecord.from_execution(
            exec_rec, success=True, score=1.0, evaluated_at="2026-01-01T00:00:00+00:00"
        )
        metric_rec = MetricRecord(
            benchmark="mock",
            agent="test",
            evaluation_count=1,
            success_rate=1.0,
            repeated_run_consistency=1.0,
            composite_reliability=1.0,
            computed_at="2026-01-01T00:00:00+00:00",
        )
        ranking_rec = RankingRecord(
            ranking_type="success",
            benchmark="mock",
            rankings=[("test", 1.0)],
            rank_map={"test": 1},
            computed_at="2026-01-01T00:00:00+00:00",
        )
        original = ExperimentResult(
            configuration=config,
            execution_records=[exec_rec],
            evaluation_records=[eval_rec],
            metric_records=[metric_rec],
            ranking_records=[ranking_rec],
        )
        key = config.sha256()
        backend.set(key, original)
        loaded = backend.get(key)
        assert loaded is not None
        assert len(loaded.execution_records) == 1
        assert len(loaded.evaluation_records) == 1
        assert len(loaded.metric_records) == 1
        assert len(loaded.ranking_records) == 1
        assert loaded.execution_records[0].task_id == "t1"
        assert loaded.metric_records[0].success_rate == 1.0

    def test_canonical_json_output(self, config: Configuration):
        result = ExperimentResult(
            configuration=config,
            execution_records=[],
            evaluation_records=[],
            metric_records=[],
            ranking_records=[],
        )
        json_str = result.canonical_json()
        parsed = json.loads(json_str)
        assert "configuration" in parsed
        assert "execution_records" in parsed

    def test_from_canonical_json(self, config: Configuration):
        result = ExperimentResult(
            configuration=config,
            execution_records=[],
            evaluation_records=[],
            metric_records=[],
            ranking_records=[],
        )
        json_str = result.canonical_json()
        restored = ExperimentResult.from_canonical_json(json_str)
        assert restored.configuration.experiment_name == config.experiment_name


class TestCacheWithNoneValues:
    def test_set_get_with_empty_result(
        self, backend: FileSystemCacheBackend, config: Configuration
    ):
        key = config.sha256()
        result = ExperimentResult(
            configuration=config,
            execution_records=[],
            evaluation_records=[],
            metric_records=[],
            ranking_records=[],
        )
        backend.set(key, result)
        loaded = backend.get(key)
        assert loaded is not None
        assert loaded.configuration == config


class TestCacheErrorHandling:
    def test_set_failure_propagates(
        self, backend: FileSystemCacheBackend, result: ExperimentResult, config: Configuration
    ):
        key = config.sha256()
        with patch.object(result.__class__, "canonical_json", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                backend.set(key, result)

    def test_get_corrupt_file_returns_none(
        self, backend: FileSystemCacheBackend, config: Configuration, result: ExperimentResult
    ):
        key = config.sha256()
        backend.set(key, result)
        path = backend._path(key)
        path.write_text("{corrupt: unparseable json}", encoding="utf-8")
        loaded = backend.get(key)
        assert loaded is None

    def test_get_nonexistent_file_returns_none(self, backend: FileSystemCacheBackend):
        assert backend.get("nonexistent-key") is None

    def test_invalidate_nonexistent_key_does_not_raise(self, backend: FileSystemCacheBackend):
        backend.invalidate("i-do-not-exist")

    def test_clear_empty_cache_does_not_raise(self, backend: FileSystemCacheBackend):
        backend.clear()

    def test_cache_backend_property(
        self, experiment_cache: ExperimentCache, backend: FileSystemCacheBackend
    ):
        assert experiment_cache.backend is backend


class TestCachePersistenceAcrossInstances:
    def test_persistence_across_backend_instances(
        self, tmp_path: Path, config: Configuration, result: ExperimentResult
    ):
        key = config.sha256()
        backend1 = FileSystemCacheBackend(cache_dir=tmp_path)
        backend1.set(key, result)

        backend2 = FileSystemCacheBackend(cache_dir=tmp_path)
        loaded = backend2.get(key)
        assert loaded is not None
        assert loaded.configuration == config

    def test_persistence_across_cache_instances(
        self, tmp_path: Path, config: Configuration, result: ExperimentResult
    ):
        key = config.sha256()
        backend = FileSystemCacheBackend(cache_dir=tmp_path)
        cache1 = ExperimentCache(backend=backend, enabled=True)
        cache1.set(key, result)

        cache2 = ExperimentCache(backend=backend, enabled=True)
        loaded = cache2.get(key)
        assert loaded is not None
        assert loaded.configuration == config

    def test_cache_cleared_between_instances(
        self, tmp_path: Path, config: Configuration, result: ExperimentResult
    ):
        key = config.sha256()
        backend = FileSystemCacheBackend(cache_dir=tmp_path)
        cache1 = ExperimentCache(backend=backend, enabled=True)
        cache1.set(key, result)
        cache1.clear()

        cache2 = ExperimentCache(backend=backend, enabled=True)
        assert cache2.get(key) is None
