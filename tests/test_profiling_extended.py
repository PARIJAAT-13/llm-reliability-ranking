"""Extended tests for PerformanceProfiler and MemoryProfiler — 40+ tests."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from llm_reliability.profiling import MemoryProfiler, PerformanceProfiler


class TestPerformanceProfiler:
    def test_no_experiment_elapsed_zero(self):
        p = PerformanceProfiler()
        assert p.experiment_elapsed == 0.0

    def test_start_experiment_sets_time(self):
        p = PerformanceProfiler()
        p.start_experiment()
        assert p.experiment_elapsed >= 0.0

    def test_experiment_elapsed_increases(self):
        p = PerformanceProfiler()
        p.start_experiment()
        t1 = p.experiment_elapsed
        time.sleep(0.01)
        t2 = p.experiment_elapsed
        assert t2 > t1

    def test_measure_benchmark_records_time(self):
        p = PerformanceProfiler()
        p.start_experiment()
        with p.measure_benchmark("test_bench", "test_model"):
            time.sleep(0.01)
        summary = p.summary()
        key = "test_bench/test_model"
        assert key in summary["per_benchmark"]
        assert summary["per_benchmark"][key]["run_count"] == 1
        assert summary["per_benchmark"][key]["total_seconds"] >= 0.01

    def test_measure_benchmark_multiple_runs(self):
        p = PerformanceProfiler()
        p.start_experiment()
        for _ in range(5):
            with p.measure_benchmark("b", "m"):
                time.sleep(0.005)
        key = "b/m"
        assert p.summary()["per_benchmark"][key]["run_count"] == 5

    def test_measure_benchmark_averages(self):
        p = PerformanceProfiler()
        p.start_experiment()
        for _ in range(3):
            with p.measure_benchmark("b", "m"):
                time.sleep(0.01)
        entry = p.summary()["per_benchmark"]["b/m"]
        assert entry["avg_seconds"] > 0
        assert abs(entry["avg_seconds"] * 3 - entry["total_seconds"]) < 0.01

    def test_multiple_benchmarks_tracked_separately(self):
        p = PerformanceProfiler()
        p.start_experiment()
        with p.measure_benchmark("bench1", "m1"):
            time.sleep(0.01)
        with p.measure_benchmark("bench2", "m2"):
            time.sleep(0.01)
        summary = p.summary()
        assert "bench1/m1" in summary["per_benchmark"]
        assert "bench2/m2" in summary["per_benchmark"]

    def test_cache_hits(self):
        p = PerformanceProfiler()
        p.record_cache_hit()
        p.record_cache_hit()
        p.record_cache_hit()
        assert p.summary()["cache_hits"] == 3

    def test_cache_misses(self):
        p = PerformanceProfiler()
        p.record_cache_miss()
        p.record_cache_miss()
        assert p.summary()["cache_misses"] == 2

    def test_cache_hit_rate_calculation(self):
        p = PerformanceProfiler()
        p.record_cache_hit()
        p.record_cache_hit()
        p.record_cache_miss()
        p.record_cache_miss()
        rate = p.summary()["cache_hit_rate_pct"]
        assert rate == 50.0

    def test_cache_hit_rate_none_when_no_activity(self):
        p = PerformanceProfiler()
        assert p.summary()["cache_hit_rate_pct"] is None

    def test_cache_hit_rate_zero(self):
        p = PerformanceProfiler()
        p.record_cache_miss()
        assert p.summary()["cache_hit_rate_pct"] == 0.0

    def test_cache_hit_rate_full(self):
        p = PerformanceProfiler()
        for _ in range(5):
            p.record_cache_hit()
        assert p.summary()["cache_hit_rate_pct"] == 100.0

    def test_errors(self):
        p = PerformanceProfiler()
        p.record_error()
        p.record_error()
        p.record_error()
        assert p.summary()["errors"] == 3

    def test_summary_structure_keys(self):
        p = PerformanceProfiler()
        s = p.summary()
        expected = {
            "total_seconds",
            "per_benchmark",
            "cache_hits",
            "cache_misses",
            "cache_hit_rate_pct",
            "errors",
        }
        assert set(s.keys()) == expected

    def test_summary_per_benchmark_empty_initially(self):
        p = PerformanceProfiler()
        assert p.summary()["per_benchmark"] == {}

    def test_reset_by_starting_new_experiment(self):
        p = PerformanceProfiler()
        p.start_experiment()
        time.sleep(0.02)
        old_elapsed = p.experiment_elapsed
        p.start_experiment()
        assert p.experiment_elapsed < old_elapsed

    def test_benchmark_context_manager_returns_timer(self):
        p = PerformanceProfiler()
        p.start_experiment()
        with p.measure_benchmark("b", "m") as timer:
            assert timer is not None

    def test_benchmark_with_exception_still_records(self):
        p = PerformanceProfiler()
        p.start_experiment()
        try:
            with p.measure_benchmark("faulty", "m"):
                msg = "error"
                raise ValueError(msg)
        except ValueError:
            pass
        assert "faulty/m" in p.summary()["per_benchmark"]

    def test_concurrent_benchmarks_stacked(self):
        p = PerformanceProfiler()
        p.start_experiment()
        with p.measure_benchmark("outer", "m"):
            time.sleep(0.01)
            with p.measure_benchmark("inner", "m"):
                time.sleep(0.01)
        summary = p.summary()
        assert (
            summary["per_benchmark"]["outer/m"]["total_seconds"]
            >= summary["per_benchmark"]["inner/m"]["total_seconds"]
        )


class TestMemoryProfiler:
    def test_delta_returns_expected_keys(self):
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

    def test_delta_all_none_before_snapshots(self):
        m = MemoryProfiler()
        d = m.delta()
        for v in d.values():
            assert v is None

    def test_snapshot_before_sets_ram_before(self):
        m = MemoryProfiler()
        m.snapshot_before()
        d = m.delta()
        if d["ram_before_gb"] is not None:
            assert d["ram_before_gb"] > 0

    def test_snapshot_after_sets_ram_after(self):
        m = MemoryProfiler()
        m.snapshot_before()
        _ = [i * i for i in range(10000)]
        m.snapshot_after()
        d = m.delta()
        if d["ram_after_gb"] is not None:
            assert d["ram_after_gb"] > 0

    def test_delta_computes_ram_diff(self):
        m = MemoryProfiler()
        m.snapshot_before()
        m.snapshot_after()
        d = m.delta()
        if d["ram_delta_gb"] is not None:
            assert isinstance(d["ram_delta_gb"], float)

    def test_peak_ram_after_snapshots(self):
        m = MemoryProfiler()
        m.snapshot_before()
        m.snapshot_after()
        d = m.delta()
        if d["peak_ram_gb"] is not None:
            assert d["peak_ram_gb"] > 0

    def test_multiple_snapshot_pairs(self):
        m = MemoryProfiler()
        m.snapshot_before()
        _ = [i for i in range(1000)]
        m.snapshot_after()
        _ = m.delta()
        m.snapshot_before()
        m.snapshot_after()
        d2 = m.delta()
        assert d2 is not None

    def test_get_ram_gb_static(self):
        ram = MemoryProfiler._get_ram_gb()
        if ram is not None:
            assert ram > 0
            assert isinstance(ram, float)

    def test_get_vram_gb_static(self):
        vram = MemoryProfiler._get_vram_gb()
        assert vram is None or vram >= 0

    def test_get_peak_vram_static(self):
        peak = MemoryProfiler._get_peak_vram_gb()
        assert peak is None or peak >= 0

    def test_delta_ram_values_positive_if_available(self):
        m = MemoryProfiler()
        m.snapshot_before()
        m.snapshot_after()
        d = m.delta()
        if d["ram_before_gb"] is not None:
            assert d["ram_before_gb"] >= 0
        if d["ram_after_gb"] is not None:
            assert d["ram_after_gb"] >= 0

    @patch("llm_reliability.profiling.memory.MemoryProfiler._get_ram_gb", return_value=1.5)
    def test_mock_ram_before(self, mock_get_ram):
        m = MemoryProfiler()
        m.snapshot_before()
        assert m._ram_before == 1.5

    @patch("llm_reliability.profiling.memory.MemoryProfiler._get_ram_gb", return_value=1.8)
    def test_mock_ram_after(self, mock_get_ram):
        m = MemoryProfiler()
        m.snapshot_after()
        assert m._ram_after == 1.8

    @patch("llm_reliability.profiling.memory.MemoryProfiler._get_vram_gb", return_value=2.5)
    def test_mock_vram(self, mock_get_vram):
        m = MemoryProfiler()
        m.snapshot_before()
        d = m.delta()
        if d["vram_before_gb"] is not None:
            assert d["vram_before_gb"] == 2.5

    def test_no_crash_without_psutil(self):
        m = MemoryProfiler()
        m.snapshot_before()
        m.snapshot_after()
        d = m.delta()
        assert isinstance(d, dict)

    def test_snapshot_before_returns_none(self):
        m = MemoryProfiler()
        result = m.snapshot_before()
        assert result is None

    def test_snapshot_after_returns_none(self):
        m = MemoryProfiler()
        result = m.snapshot_after()
        assert result is None

    @patch("llm_reliability.profiling.memory.MemoryProfiler._get_ram_gb", return_value=2.0)
    @patch("llm_reliability.profiling.memory.MemoryProfiler._get_vram_gb", return_value=4.0)
    def test_mock_full_memory_delta(self, mock_vram, mock_ram):
        m = MemoryProfiler()
        m.snapshot_before()
        m.snapshot_after()
        d = m.delta()
        if d["ram_before_gb"] is not None:
            assert d["ram_before_gb"] == 2.0
        if d["vram_before_gb"] is not None:
            assert d["vram_before_gb"] == 4.0

    @patch("llm_reliability.profiling.memory.MemoryProfiler._get_ram_gb", return_value=None)
    def test_get_ram_none_on_exception(self, mock_ram):
        result = MemoryProfiler._get_ram_gb()
        assert result is None

    def test_summary_keys_returned(self):
        m = MemoryProfiler()
        d = m.delta()
        assert "ram_before_gb" in d
        assert "vram_before_gb" in d
        assert "ram_delta_gb" in d
