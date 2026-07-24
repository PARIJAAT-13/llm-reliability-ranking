"""Tests for performance and memory profiling."""

from __future__ import annotations

import time

import pytest

from llm_reliability.profiling import MemoryProfiler, PerformanceProfiler


class TestPerformanceProfiler:
    def test_start_experiment(self):
        p = PerformanceProfiler()
        p.start_experiment()
        assert p.experiment_elapsed >= 0

    def test_measure_benchmark(self):
        p = PerformanceProfiler()
        p.start_experiment()
        with p.measure_benchmark("agentboard", "gpt-4"):
            time.sleep(0.01)
        summary = p.summary()
        assert "agentboard/gpt-4" in summary["per_benchmark"]
        entry = summary["per_benchmark"]["agentboard/gpt-4"]
        assert entry["benchmark"] == "agentboard"
        assert entry["model"] == "gpt-4"
        assert entry["total_seconds"] >= 0.01
        assert entry["run_count"] == 1

    def test_multiple_benchmarks(self):
        p = PerformanceProfiler()
        p.start_experiment()
        with p.measure_benchmark("bench1", "model-a"):
            time.sleep(0.005)
        with p.measure_benchmark("bench2", "model-b"):
            time.sleep(0.005)
        with p.measure_benchmark("bench1", "model-a"):
            time.sleep(0.005)
        summary = p.summary()
        assert summary["per_benchmark"]["bench1/model-a"]["run_count"] == 2
        assert summary["per_benchmark"]["bench2/model-b"]["run_count"] == 1

    def test_cache_hit_miss(self):
        p = PerformanceProfiler()
        p.record_cache_hit()
        p.record_cache_hit()
        p.record_cache_miss()
        summary = p.summary()
        assert summary["cache_hits"] == 2
        assert summary["cache_misses"] == 1
        assert summary["cache_hit_rate_pct"] == pytest.approx(66.7, rel=0.1)

    def test_cache_hit_rate_none_when_no_data(self):
        p = PerformanceProfiler()
        summary = p.summary()
        assert summary["cache_hit_rate_pct"] is None

    def test_errors(self):
        p = PerformanceProfiler()
        p.record_error()
        p.record_error()
        assert p.summary()["errors"] == 2

    def test_empty_summary(self):
        p = PerformanceProfiler()
        s = p.summary()
        assert s["total_seconds"] >= 0
        assert s["per_benchmark"] == {}
        assert s["cache_hits"] == 0
        assert s["cache_misses"] == 0


class TestMemoryProfiler:
    def test_snapshot_before_and_after(self):
        m = MemoryProfiler()
        m.snapshot_before()
        [i * i for i in range(1000)]
        m.snapshot_after()
        d = m.delta()
        # RAM values may be None on systems without psutil
        if d["ram_before_gb"] is not None:
            assert d["ram_before_gb"] > 0
            assert d["ram_after_gb"] > 0

    def test_delta_returns_expected_keys(self):
        m = MemoryProfiler()
        d = m.delta()
        expected_keys = {
            "ram_before_gb",
            "ram_after_gb",
            "ram_delta_gb",
            "vram_before_gb",
            "vram_after_gb",
            "vram_delta_gb",
            "peak_ram_gb",
            "peak_vram_gb",
        }
        assert set(d.keys()) == expected_keys

    def test_null_when_no_snapshots(self):
        m = MemoryProfiler()
        d = m.delta()
        assert d["ram_before_gb"] is None
        assert d["ram_after_gb"] is None

    def test_vram_graceful_degradation(self):
        m = MemoryProfiler()
        m.snapshot_before()
        m.snapshot_after()
        d = m.delta()
        # On systems without CUDA, vram values are None
        assert d["vram_before_gb"] is None or d["vram_before_gb"] >= 0
