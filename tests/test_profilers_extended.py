"""Extended tests for PerformanceProfiler and MemoryProfiler."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from llm_reliability.profiling import MemoryProfiler, PerformanceProfiler

# ======================================================================
# PerformanceProfiler
# ======================================================================


class TestPerformanceProfilerExtended:
    def test_perf_profiler_no_experiment(self):
        p = PerformanceProfiler()
        assert p.experiment_elapsed == 0.0
        assert p.summary()["total_seconds"] == 0.0

    def test_perf_profiler_double_start(self):
        p = PerformanceProfiler()
        p.start_experiment()
        t1 = p.experiment_elapsed
        p.start_experiment()
        t2 = p.experiment_elapsed
        assert t2 < t1 + 0.01

    def test_perf_profiler_measure_benchmark_nested(self):
        p = PerformanceProfiler()
        p.start_experiment()
        with p.measure_benchmark("outer", "model-a"):
            with p.measure_benchmark("inner", "model-b"):
                time.sleep(0.02)
        summary = p.summary()
        assert "outer/model-a" in summary["per_benchmark"]
        assert "inner/model-b" in summary["per_benchmark"]
        assert (
            summary["per_benchmark"]["outer/model-a"]["total_seconds"]
            >= summary["per_benchmark"]["inner/model-b"]["total_seconds"]
        )

    def test_perf_profiler_benchmark_with_error(self):
        p = PerformanceProfiler()
        p.start_experiment()
        try:
            with p.measure_benchmark("faulty", "model-x"):
                raise RuntimeError("benchmark crashed")
        except RuntimeError:
            pass
        summary = p.summary()
        assert "faulty/model-x" in summary["per_benchmark"]
        assert summary["per_benchmark"]["faulty/model-x"]["run_count"] == 1
        assert summary["per_benchmark"]["faulty/model-x"]["total_seconds"] >= 0

    def test_perf_profiler_empty_summary_structure(self):
        p = PerformanceProfiler()
        s = p.summary()
        expected_keys = {
            "total_seconds",
            "per_benchmark",
            "cache_hits",
            "cache_misses",
            "cache_hit_rate_pct",
            "errors",
        }
        assert set(s.keys()) == expected_keys
        assert s["per_benchmark"] == {}
        assert s["cache_hits"] == 0
        assert s["cache_misses"] == 0
        assert s["errors"] == 0

    def test_perf_profiler_multiple_experiments(self):
        p = PerformanceProfiler()

        p.start_experiment()
        time.sleep(0.02)
        assert p.experiment_elapsed >= 0.01
        with p.measure_benchmark("bench1", "m1"):
            time.sleep(0.02)

        p.start_experiment()
        new_elapsed = p.experiment_elapsed
        assert new_elapsed < 0.01
        with p.measure_benchmark("bench2", "m2"):
            time.sleep(0.02)

        s = p.summary()
        assert "bench1/m1" in s["per_benchmark"]
        assert "bench2/m2" in s["per_benchmark"]
        assert s["per_benchmark"]["bench1/m1"]["run_count"] == 1
        assert s["per_benchmark"]["bench2/m2"]["run_count"] == 1


# ======================================================================
# MemoryProfiler
# ======================================================================


class TestMemoryProfilerExtended:
    def test_memory_profiler_no_psutil(self):
        m = MemoryProfiler()
        m._get_ram_gb = staticmethod(lambda: None)
        m._get_vram_gb = staticmethod(lambda: None)
        m._get_peak_vram_gb = staticmethod(lambda: None)
        m.snapshot_before()
        m.snapshot_after()
        d = m.delta()
        assert d["ram_before_gb"] is None
        assert d["ram_after_gb"] is None
        assert d["ram_delta_gb"] is None

    def test_memory_profiler_no_torch(self):
        m = MemoryProfiler()
        m._get_vram_gb = staticmethod(lambda: None)
        m._get_peak_vram_gb = staticmethod(lambda: None)
        m.snapshot_before()
        m.snapshot_after()
        d = m.delta()
        assert d["vram_before_gb"] is None
        assert d["vram_after_gb"] is None
        assert d["vram_delta_gb"] is None
        assert d["peak_vram_gb"] is None

    def test_memory_profiler_cpu_only(self):
        m = MemoryProfiler()
        m._get_vram_gb = staticmethod(lambda: None)
        m._get_peak_vram_gb = staticmethod(lambda: None)
        m.snapshot_before()
        m.snapshot_after()
        d = m.delta()
        assert d["vram_before_gb"] is None
        assert d["vram_after_gb"] is None
        if d["ram_before_gb"] is not None:
            assert d["ram_before_gb"] > 0

    def test_memory_profiler_delta_keys_structure(self):
        m = MemoryProfiler()
        d = m.delta()
        expected = {
            "ram_before_gb",
            "ram_after_gb",
            "ram_delta_gb",
            "vram_before_gb",
            "vram_after_gb",
            "vram_delta_gb",
            "peak_ram_gb",
            "peak_vram_gb",
        }
        assert set(d.keys()) == expected

    def test_memory_profiler_double_snapshot(self):
        m = MemoryProfiler()
        m.snapshot_before()
        m.snapshot_before()
        m.snapshot_after()
        m.snapshot_after()
        d = m.delta()
        assert d["ram_before_gb"] is None or d["ram_before_gb"] > 0
        assert d["ram_after_gb"] is None or d["ram_after_gb"] > 0

    def test_memory_profiler_delta_without_before(self):
        m = MemoryProfiler()
        m.snapshot_after()
        d = m.delta()
        assert d["ram_before_gb"] is None
        assert d["ram_delta_gb"] is None

    def test_memory_profiler_delta_without_after(self):
        m = MemoryProfiler()
        m.snapshot_before()
        d = m.delta()
        assert d["ram_after_gb"] is None
        assert d["ram_delta_gb"] is None

    def test_memory_profiler_snapshot_returns_expected_fields(self):
        m = MemoryProfiler()
        m.snapshot_before()
        assert m._ram_before is None or m._ram_before > 0
        assert m._vram_before is None or m._vram_before == 0 or m._vram_before > 0
        m.snapshot_after()
        assert m._ram_after is None or m._ram_after > 0
        assert m._vram_after is None or m._vram_after == 0 or m._vram_after > 0
        if m._ram_before is not None and m._ram_after is not None:
            assert m._peak_ram == max(m._ram_before, m._ram_after)

    def test_memory_profiler_no_psutil_no_torch(self):
        m = MemoryProfiler()
        m._get_ram_gb = staticmethod(lambda: None)
        m._get_vram_gb = staticmethod(lambda: None)
        m._get_peak_vram_gb = staticmethod(lambda: None)
        m.snapshot_before()
        m.snapshot_after()
        d = m.delta()
        for key in d:
            assert d[key] is None

    def test_memory_profiler_mocked_values(self):
        m = MemoryProfiler()
        m._get_ram_gb = staticmethod(lambda: 2.5)
        m._get_vram_gb = staticmethod(lambda: 1.2)
        m._get_peak_vram_gb = staticmethod(lambda: 1.5)
        m.snapshot_before()
        m._get_ram_gb = staticmethod(lambda: 3.0)
        m._get_vram_gb = staticmethod(lambda: 1.8)
        m._get_peak_vram_gb = staticmethod(lambda: 1.8)
        m.snapshot_after()
        d = m.delta()
        assert d["ram_before_gb"] == 2.5
        assert d["ram_after_gb"] == 3.0
        assert d["ram_delta_gb"] == 0.5
        assert d["vram_before_gb"] == 1.2
        assert d["vram_after_gb"] == 1.8
        assert d["vram_delta_gb"] == 0.6
        assert d["peak_ram_gb"] == 3.0
        assert d["peak_vram_gb"] == 1.8


# ======================================================================
# Combined / Cache tests
# ======================================================================


class TestProfilerCaches:
    def test_profiler_caches_hit_miss_counts(self):
        p = PerformanceProfiler()
        for _ in range(5):
            p.record_cache_hit()
        for _ in range(3):
            p.record_cache_miss()
        p.record_error()
        s = p.summary()
        assert s["cache_hits"] == 5
        assert s["cache_misses"] == 3
        assert s["cache_hit_rate_pct"] == pytest.approx(62.5, rel=0.1)
        assert s["errors"] == 1

    def test_profiler_cache_hit_rate_100(self):
        p = PerformanceProfiler()
        p.record_cache_hit()
        s = p.summary()
        assert s["cache_hit_rate_pct"] == 100.0

    def test_profiler_cache_hit_rate_0(self):
        p = PerformanceProfiler()
        p.record_cache_miss()
        s = p.summary()
        assert s["cache_hit_rate_pct"] == 0.0

    def test_profiler_cache_hit_rate_none(self):
        p = PerformanceProfiler()
        s = p.summary()
        assert s["cache_hit_rate_pct"] is None
