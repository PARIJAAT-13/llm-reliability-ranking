"""Tests for Cloud Baseline Engine."""

from decimal import Decimal

import pytest

from llm_reliability.cloud_baseline.engine import (CLOUD_PROVIDERS,
                                                   CloudBaselineEngine)
from llm_reliability.cloud_baseline.models import (CloudBaselineComparison,
                                                   CloudBaselineResult,
                                                   CloudBaselineSummary)
from llm_reliability.cloud_baseline.report import CloudBaselineReportGenerator
from llm_reliability.configs.config import Configuration


def test_cloud_baseline_result_creation():
    r = CloudBaselineResult(
        provider="openai",
        model="gpt-4o",
        benchmark="GAIA",
        task_id="g1",
        success=True,
        score=1.0,
        cost_usd=Decimal("0.0025"),
        latency_ms=1200.0,
        tokens_input=100,
        tokens_output=50,
        runtime_seconds=1.5,
    )
    assert r.provider == "openai"
    assert r.score == 1.0
    assert r.cost_usd == Decimal("0.0025")


def test_cloud_baseline_summary_compute_all():
    results = [
        CloudBaselineResult(
            provider="openai",
            model="gpt-4o",
            benchmark="GAIA",
            task_id="g1",
            success=True,
            score=1.0,
            cost_usd=Decimal("0.001"),
            latency_ms=100.0,
            tokens_input=10,
            tokens_output=20,
            runtime_seconds=1.0,
        ),
        CloudBaselineResult(
            provider="openai",
            model="gpt-4o",
            benchmark="GAIA",
            task_id="g2",
            success=False,
            score=0.0,
            cost_usd=Decimal("0.002"),
            latency_ms=200.0,
            tokens_input=15,
            tokens_output=30,
            runtime_seconds=2.0,
        ),
        CloudBaselineResult(
            provider="anthropic",
            model="claude-3-5-sonnet",
            benchmark="GAIA",
            task_id="g1",
            success=True,
            score=1.0,
            cost_usd=Decimal("0.003"),
            latency_ms=150.0,
            tokens_input=10,
            tokens_output=20,
            runtime_seconds=1.5,
        ),
    ]

    summaries = CloudBaselineSummary.compute_all(results)
    assert len(summaries) == 2

    openai_summary = next(s for s in summaries if s.provider == "openai")
    assert openai_summary.success_rate == 0.5
    assert openai_summary.total_cost_usd == Decimal("0.003")
    assert openai_summary.task_count == 2

    anthro_summary = next(s for s in summaries if s.provider == "anthropic")
    assert anthro_summary.success_rate == 1.0
    assert anthro_summary.task_count == 1


def test_cloud_baseline_comparison():
    summaries = [
        CloudBaselineSummary(
            provider="openai",
            model="gpt-4o",
            benchmark="GAIA",
            total_cost_usd=Decimal("0.01"),
            avg_latency_ms=100.0,
            p50_latency_ms=100.0,
            p95_latency_ms=150.0,
            p99_latency_ms=200.0,
            success_rate=0.8,
            total_tokens=300,
            task_count=10,
            cost_per_task_usd=Decimal("0.001"),
            cost_per_success_usd=Decimal("0.00125"),
        ),
        CloudBaselineSummary(
            provider="anthropic",
            model="claude-3-5-sonnet",
            benchmark="GAIA",
            total_cost_usd=Decimal("0.02"),
            avg_latency_ms=200.0,
            p50_latency_ms=200.0,
            p95_latency_ms=300.0,
            p99_latency_ms=400.0,
            success_rate=0.9,
            total_tokens=500,
            task_count=10,
            cost_per_task_usd=Decimal("0.002"),
            cost_per_success_usd=Decimal("0.00222"),
        ),
    ]

    comp = CloudBaselineComparison.compute(summaries)
    assert comp.benchmark == "GAIA"
    assert comp.best_provider == "anthropic"
    assert comp.best_score == 0.9
    assert comp.most_efficient_provider == "openai"


def test_cloud_baseline_empty_results():
    summaries = CloudBaselineSummary.compute_all([])
    assert summaries == []

    comp = CloudBaselineComparison.compute([])
    assert comp is None


def test_cloud_baseline_latency_percentiles():
    results = [
        CloudBaselineResult(
            provider="openai",
            model="gpt-4o",
            benchmark="B",
            task_id=f"t{i}",
            success=True,
            score=1.0,
            cost_usd=Decimal("0.001"),
            latency_ms=float(100 + i * 10),
            tokens_input=10,
            tokens_output=20,
            runtime_seconds=1.0,
        )
        for i in range(10)
    ]
    summaries = CloudBaselineSummary.compute_all(results)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.p50_latency_ms >= s.p50_latency_ms - 1
    assert s.p95_latency_ms >= s.p50_latency_ms


def test_cloud_baseline_report_generation():
    results = [
        CloudBaselineResult(
            provider="openai",
            model="gpt-4o",
            benchmark="GAIA",
            task_id="g1",
            success=True,
            score=1.0,
            cost_usd=Decimal("0.001"),
            latency_ms=100.0,
            tokens_input=10,
            tokens_output=20,
            runtime_seconds=1.0,
        ),
    ]
    report = CloudBaselineReportGenerator.summary_table(results)
    assert "# Cloud Baseline Comparison" in report
    assert "openai" in report
    assert "gpt-4o" in report

    csv = CloudBaselineReportGenerator.csv_report(results)
    assert "provider,model" in csv
    assert "openai,gpt-4o" in csv


def test_cloud_baseline_detail_report_with_errors():
    results = [
        CloudBaselineResult(
            provider="openai",
            model="gpt-4o",
            benchmark="GAIA",
            task_id="g1",
            success=False,
            score=0.0,
            cost_usd=Decimal("0"),
            latency_ms=0.0,
            tokens_input=0,
            tokens_output=0,
            runtime_seconds=0.0,
            error="API timeout",
        ),
    ]
    report = CloudBaselineReportGenerator.detailed_report(results)
    assert "ERROR: API timeout" in report


def test_cloud_providers_defined():
    assert "openai" in CLOUD_PROVIDERS
    assert "anthropic" in CLOUD_PROVIDERS
    assert "google" in CLOUD_PROVIDERS
    assert "gpt-4o" in CLOUD_PROVIDERS["openai"]
    assert "gpt-4.1" in CLOUD_PROVIDERS["openai"]


def test_cloud_baseline_engine_initialization():
    config = Configuration(
        experiment_name="test",
        benchmark="GAIA",
        agent="dummy",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": "data/gaia_sample.json"},
    )
    engine = CloudBaselineEngine(config)
    assert engine._config is not None
    assert engine.results == []


def test_cloud_baseline_engine_unknown_provider():
    engine = CloudBaselineEngine()
    with pytest.raises(ValueError, match="Unknown cloud provider"):
        engine.run_single("unknown", "model", "GAIA", {"task_id": "t1"})
