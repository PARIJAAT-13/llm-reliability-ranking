"""Tests for publication-ready reporting and reproducibility manifests."""

import json
import tempfile
from pathlib import Path

import pytest

from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord
from llm_reliability.reporting.publication import (
    generate_benchmark_summary, generate_csv, generate_experiment_summary,
    generate_latex_table, generate_markdown_table, generate_ranking_summary,
    generate_reproducibility_manifest, generate_runtime_summary,
    generate_statistics_summary, save_publication_artifacts)


def _sha64(s: str) -> str:
    return s.ljust(64, "0")[:64]


@pytest.fixture
def sample_metrics() -> list[MetricRecord]:
    return [
        MetricRecord(
            agent="model-a",
            benchmark="bench1",
            evaluation_count=10,
            success_rate=1.0,
            repeated_run_consistency=0.95,
            composite_reliability=0.95,
            computed_at="2026-01-01T00:00:00Z",
        ),
        MetricRecord(
            agent="model-b",
            benchmark="bench1",
            evaluation_count=10,
            success_rate=0.8,
            repeated_run_consistency=0.7,
            composite_reliability=0.75,
            computed_at="2026-01-01T00:00:00Z",
        ),
    ]


@pytest.fixture
def sample_rankings() -> list[RankingRecord]:
    return [
        RankingRecord(
            ranking_type="reliability",
            benchmark="bench1",
            rankings=(("model-a", 0.95), ("model-b", 0.75)),
            rank_map={"model-a": 1, "model-b": 2},
            computed_at="2026-01-01T00:00:00Z",
        ),
        RankingRecord(
            ranking_type="success",
            benchmark="bench1",
            rankings=(("model-a", 1.0), ("model-b", 0.8)),
            rank_map={"model-a": 1, "model-b": 2},
            computed_at="2026-01-01T00:00:00Z",
        ),
    ]


@pytest.fixture
def sample_executions() -> list[ExecutionRecord]:
    return [
        ExecutionRecord(
            configuration_hash=_sha64("abc"),
            seed=42,
            benchmark="bench1",
            agent="model-a",
            task_id="t1",
            run_index=0,
            runtime_seconds=2.5,
            timestamp="2026-01-01T00:00:00",
            stdout="ok",
            stderr="",
            status="success",
        ),
        ExecutionRecord(
            configuration_hash=_sha64("def"),
            seed=43,
            benchmark="bench1",
            agent="model-b",
            task_id="t2",
            run_index=0,
            runtime_seconds=5.0,
            timestamp="2026-01-01T00:00:01",
            stdout="ok",
            stderr="",
            status="error",
            error="fail",
        ),
    ]


class TestExperimentSummary:
    def test_generate(self, sample_metrics, sample_rankings, sample_executions) -> None:
        summary = generate_experiment_summary(
            experiment_id="test",
            metrics=sample_metrics,
            rankings=sample_rankings,
            executions=sample_executions,
        )
        assert summary["experiment_id"] == "test"
        assert "reliability" in summary
        assert "rankings" in summary
        assert summary["total_metrics"] == 2
        assert summary["total_executions"] == 2

    def test_empty_inputs(self) -> None:
        summary = generate_experiment_summary("empty", [], [], [])
        assert summary["total_metrics"] == 0
        assert summary["reliability"]["mean"] == 0.0


class TestRuntimeSummary:
    def test_generate(self) -> None:
        summary = generate_runtime_summary({"name": "ollama", "version": "0.1.30"})
        assert "runtime" in summary
        assert "python" in summary


class TestBenchmarkSummary:
    def test_generate(self, sample_metrics, sample_executions) -> None:
        results = generate_benchmark_summary(sample_metrics, sample_executions)
        assert len(results) == 1
        assert results[0]["benchmark"] == "bench1"
        assert results[0]["mean_reliability"] > 0


class TestRankingSummary:
    def test_generate(self, sample_rankings) -> None:
        results = generate_ranking_summary(sample_rankings)
        assert len(results) == 2
        for r in results:
            assert "ranking_type" in r
            assert "rankings" in r


class TestStatisticsSummary:
    def test_generate(self, sample_metrics) -> None:
        stats = generate_statistics_summary(sample_metrics)
        assert stats["model_count"] == 2
        assert stats["reliability"]["mean"] > 0
        assert stats["success_rate"]["mean"] > 0

    def test_empty(self) -> None:
        stats = generate_statistics_summary([])
        assert stats["model_count"] == 0


class TestLatexTable:
    def test_generate(self, sample_rankings) -> None:
        latex = generate_latex_table(sample_rankings)
        assert "\\begin{table}" in latex
        assert "model-a" in latex
        assert "\\caption{" in latex

    def test_empty(self) -> None:
        latex = generate_latex_table([])
        assert "No ranking data" in latex


class TestMarkdownTable:
    def test_generate(self, sample_rankings) -> None:
        md = generate_markdown_table(sample_rankings)
        assert "| Agent |" in md
        assert "model-a" in md

    def test_empty(self) -> None:
        md = generate_markdown_table([])
        assert "No ranking data" in md


class TestCSV:
    def test_generate(self, sample_rankings) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_csv(sample_rankings, Path(tmp) / "test.csv")
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "ranking_type" in content
            assert "model-a" in content


class TestReproducibilityManifest:
    def test_generate(self) -> None:
        manifest = generate_reproducibility_manifest(
            experiment_id="test",
            config={"benchmark": "mock"},
            seeds=[42, 43],
        )
        assert manifest.experiment_id == "test"
        assert manifest.random_seeds == [42, 43]
        assert manifest.framework_version is not None
        assert manifest.python_version != ""

    def test_serialization(self) -> None:
        manifest = generate_reproducibility_manifest(experiment_id="ser-test")
        json_str = manifest.canonical_json()
        restored_dict = json.loads(json_str)
        assert restored_dict["experiment_id"] == "ser-test"


class TestSavePublicationArtifacts:
    def test_save_all(self, sample_metrics, sample_rankings, sample_executions) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_publication_artifacts(
                experiment_id="test",
                metrics=sample_metrics,
                rankings=sample_rankings,
                executions=sample_executions,
                config={"test": True},
                output_dir=Path(tmp),
            )
            assert len(paths) >= 5
            assert paths["experiment_summary"].exists()
            assert paths["ranking_summary"].exists()
            assert paths["latex_table"].exists()
            assert paths["markdown_table"].exists()
            assert paths["manifest"].exists()
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            assert manifest["experiment_id"] == "test"

    def test_save_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_publication_artifacts(
                experiment_id="empty",
                metrics=[],
                rankings=[],
                executions=[],
                output_dir=Path(tmp) / "pub",
            )
            assert paths["experiment_summary"].exists()
