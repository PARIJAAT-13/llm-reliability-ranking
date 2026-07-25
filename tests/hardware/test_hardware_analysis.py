"""Tests for hardware analysis, reports, and visualization modules."""

import os
import tempfile
from pathlib import Path

import pytest

from llm_reliability.hardware.analysis import (
    HardwareAnalysis,
    _ram_bucket,
    _vram_bucket,
)
from llm_reliability.hardware.reports import (
    generate_hardware_report,
    generate_hardware_statistics,
    generate_hardware_summary,
    save_hardware_artifacts,
)
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord
from llm_reliability.utils.hardware_profile import HardwareProfile, HardwareRegistry


def _sha64(s: str) -> str:
    return s.ljust(64, "0")[:64]


@pytest.fixture
def sample_profile() -> HardwareProfile:
    return HardwareProfile(
        profile_id="test-machine",
        profile_name="Test Machine",
        os_name="Linux",
        os_version="Ubuntu 22.04",
        cpu_architecture="x86_64",
        cpu_cores_logical=8,
        cpu_cores_physical=4,
        ram_total_gb=32.0,
        ram_available_gb=16.5,
        gpu_name="NVIDIA RTX 4090",
        gpu_count=1,
        vram_total_gb=24.0,
        python_version="3.11.5",
        node_type="local",
    )


@pytest.fixture
def sample_metrics() -> list[MetricRecord]:
    return [
        MetricRecord(
            agent="model-a",
            benchmark="test-bench",
            evaluation_count=10,
            success_rate=1.0,
            repeated_run_consistency=0.95,
            composite_reliability=0.95,
            computed_at="2026-01-01T00:00:00Z",
        ),
        MetricRecord(
            agent="model-b",
            benchmark="test-bench",
            evaluation_count=10,
            success_rate=0.8,
            repeated_run_consistency=0.7,
            composite_reliability=0.75,
            computed_at="2026-01-01T00:00:00Z",
        ),
        MetricRecord(
            agent="model-c",
            benchmark="test-bench",
            evaluation_count=10,
            success_rate=0.5,
            repeated_run_consistency=0.4,
            composite_reliability=0.45,
            computed_at="2026-01-01T00:00:00Z",
        ),
    ]


@pytest.fixture
def sample_executions() -> list[ExecutionRecord]:
    return [
        ExecutionRecord(
            configuration_hash=_sha64("abc123"),
            seed=42,
            benchmark="test-bench",
            agent="model-a",
            task_id="t1",
            run_index=0,
            runtime_seconds=2.5,
            timestamp="2026-01-01T00:00:00",
            stdout="output",
            stderr="",
            status="success",
            environment_metadata={
                "hardware_profile": "test-machine",
                "ram_total_gb": 32.0,
                "vram_total_gb": 24.0,
            },
        ),
        ExecutionRecord(
            configuration_hash=_sha64("abc124"),
            seed=43,
            benchmark="test-bench",
            agent="model-b",
            task_id="t2",
            run_index=0,
            runtime_seconds=5.0,
            timestamp="2026-01-01T00:00:01",
            stdout="output",
            stderr="",
            status="success",
            environment_metadata={
                "hardware_profile": "test-machine",
                "ram_total_gb": 16.0,
                "vram_total_gb": 8.0,
            },
        ),
        ExecutionRecord(
            configuration_hash=_sha64("abc125"),
            seed=44,
            benchmark="test-bench",
            agent="model-c",
            task_id="t3",
            run_index=0,
            runtime_seconds=10.0,
            timestamp="2026-01-01T00:00:02",
            stdout="",
            stderr="error",
            status="error",
            error="failure",
            environment_metadata={
                "hardware_profile": "other-machine",
                "ram_total_gb": 8.0,
                "vram_total_gb": 0.0,
            },
        ),
    ]


class TestHardwareAnalysis:
    def test_reliability_by_ram(self, sample_metrics, sample_executions) -> None:
        analysis = HardwareAnalysis()
        results = analysis.reliability_by_ram(sample_metrics, sample_executions)
        assert len(results) > 0
        for r in results:
            assert "ram_bucket" in r
            assert "model_count" in r
            assert "mean_reliability" in r

    def test_reliability_by_vram(self, sample_metrics, sample_executions) -> None:
        analysis = HardwareAnalysis()
        results = analysis.reliability_by_vram(sample_metrics, sample_executions)
        assert len(results) > 0
        for r in results:
            assert "vram_bucket" in r
            assert "model_count" in r

    def test_latency_by_ram(self, sample_executions) -> None:
        analysis = HardwareAnalysis()
        results = analysis.latency_by_ram(sample_executions)
        assert len(results) > 0
        for r in results:
            assert "mean_latency_seconds" in r
            assert "median_latency_seconds" in r

    def test_failure_rate_by_memory(self, sample_executions) -> None:
        analysis = HardwareAnalysis()
        results = analysis.failure_rate_by_memory(sample_executions)
        assert len(results) > 0
        for r in results:
            assert "failure_rate" in r
            assert 0.0 <= r["failure_rate"] <= 1.0

    def test_success_rate_by_hardware(self, sample_metrics, sample_executions) -> None:
        analysis = HardwareAnalysis()
        results = analysis.success_rate_by_hardware(sample_metrics, sample_executions)
        assert len(results) > 0
        for r in results:
            assert "hardware_profile" in r
            assert "mean_success_rate" in r

    def test_memory_by_model(self, sample_executions) -> None:
        analysis = HardwareAnalysis()
        results = analysis.memory_by_model(sample_executions)
        assert len(results) > 0
        for r in results:
            assert "model" in r
            assert "mean_ram_gb" in r

    def test_model_ranking_by_hardware(self, sample_metrics, sample_executions) -> None:
        analysis = HardwareAnalysis()
        results = analysis.model_ranking_by_hardware(sample_metrics, sample_executions)
        assert len(results) > 0
        for profile, rankings in results.items():
            for r in rankings:
                assert "rank" in r
                assert "model" in r
                assert "composite_reliability" in r

    def test_empty_inputs(self) -> None:
        analysis = HardwareAnalysis()
        assert analysis.reliability_by_ram([], []) == []
        assert analysis.reliability_by_vram([], []) == []
        assert analysis.latency_by_ram([]) == []
        assert analysis.failure_rate_by_memory([]) == []
        assert analysis.success_rate_by_hardware([], []) == []
        assert analysis.memory_by_model([]) == []
        assert analysis.model_ranking_by_hardware([], []) == {}


class TestBucketFunctions:
    def test_ram_bucket(self) -> None:
        assert _ram_bucket(4.0) == "0-8GB"
        assert _ram_bucket(8.0) == "8-16GB"
        assert _ram_bucket(15.0) == "8-16GB"
        assert _ram_bucket(16.0) == "16-32GB"
        assert _ram_bucket(31.0) == "16-32GB"
        assert _ram_bucket(32.0) == "32-64GB"
        assert _ram_bucket(63.0) == "32-64GB"
        assert _ram_bucket(64.0) == "64GB+"
        assert _ram_bucket(128.0) == "64GB+"

    def test_vram_bucket(self) -> None:
        assert _vram_bucket(0.0) == "no-gpu"
        assert _vram_bucket(-1.0) == "no-gpu"
        assert _vram_bucket(4.0) == "0-6GB"
        assert _vram_bucket(6.0) == "6-12GB"
        assert _vram_bucket(11.0) == "6-12GB"
        assert _vram_bucket(12.0) == "12-24GB"
        assert _vram_bucket(23.0) == "12-24GB"
        assert _vram_bucket(24.0) == "24GB+"
        assert _vram_bucket(48.0) == "24GB+"


class TestHardwareReports:
    def test_generate_hardware_summary(
        self, sample_profile, sample_metrics, sample_executions
    ) -> None:
        summary = generate_hardware_summary(sample_profile, sample_metrics, sample_executions)
        assert summary["profile_id"] == "test-machine"
        assert summary["profile_name"] == "Test Machine"
        assert summary["ram_total_gb"] == 32.0
        assert "avg_reliability" in summary
        assert "total_metrics" in summary
        assert "total_executions" in summary

    def test_generate_hardware_summary_no_data(self, sample_profile) -> None:
        summary = generate_hardware_summary(sample_profile)
        assert summary["profile_id"] == "test-machine"
        assert "avg_reliability" not in summary
        assert "total_executions" not in summary

    def test_generate_hardware_statistics(self, sample_metrics, sample_executions) -> None:
        stats = generate_hardware_statistics(sample_metrics, sample_executions)
        assert "reliability_by_ram" in stats
        assert "reliability_by_vram" in stats
        assert "latency_by_ram" in stats
        assert "failure_rate_by_memory" in stats
        assert "success_rate_by_hardware" in stats
        assert "memory_by_model" in stats
        assert "model_ranking_by_hardware" in stats

    def test_generate_hardware_report_with_data(
        self, sample_profile, sample_metrics, sample_executions
    ) -> None:
        report = generate_hardware_report(sample_profile, sample_metrics, sample_executions)
        assert "Test Machine" in report
        assert "System Overview" in report
        assert "Experiment Summary" in report
        assert "Execution Summary" in report
        assert report.endswith("*")

    def test_generate_hardware_report_no_data(self, sample_profile) -> None:
        report = generate_hardware_report(sample_profile)
        assert "System Overview" in report
        assert "Experiment Summary" not in report
        assert "Execution Summary" not in report

    def test_save_hardware_artifacts(
        self, sample_profile, sample_metrics, sample_executions
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "hw"
            paths = save_hardware_artifacts(
                sample_profile, sample_metrics, sample_executions, output_dir=out
            )
            assert paths["summary"].exists()
            assert paths["report_md"].exists()
            assert paths["report_html"].exists()
            assert paths["statistics"].exists()
            summary_text = paths["summary"].read_text(encoding="utf-8")
            assert "test-machine" in summary_text

    def test_save_hardware_artifacts_metrics_only(self, sample_profile, sample_metrics) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_hardware_artifacts(sample_profile, sample_metrics, output_dir=Path(tmp))
            assert paths["summary"].exists()
            assert paths["report_md"].exists()
            assert paths["report_html"].exists()
            assert "statistics" not in paths


class TestHardwarePackageIntegration:
    def test_with_named_profiles(self, sample_metrics, sample_executions) -> None:
        from llm_reliability.utils.hardware_profile import _register_named_profiles

        _register_named_profiles()
        assert len(HardwareRegistry.list_profiles()) >= 3
        for pid in HardwareRegistry.list_profiles():
            profile = HardwareRegistry.get(pid)
            summary = generate_hardware_summary(profile, sample_metrics, sample_executions)
            assert summary["profile_name"]
            assert summary["ram_total_gb"] > 0
