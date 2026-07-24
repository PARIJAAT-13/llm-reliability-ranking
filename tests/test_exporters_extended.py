"""Extended tests for RecordExporter — edge cases, unicode, errors, large data."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from llm_reliability.exporters import RecordExporter
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord
from tests.conftest import CONFIG_HASH, TIMESTAMP


def _make_execution(task_id: str = "task-0", **kwargs) -> ExecutionRecord:
    params = dict(
        configuration_hash=CONFIG_HASH,
        seed=42,
        benchmark="mock",
        agent="test_agent",
        task_id=task_id,
        run_index=0,
        runtime_seconds=1.0,
        timestamp=TIMESTAMP,
        stdout="ok",
        stderr="",
        status="success",
    )
    params.update(kwargs)
    return ExecutionRecord(**params)


def _make_evaluation(execution: ExecutionRecord, **kwargs) -> EvaluationRecord:
    params = dict(
        execution_hash=execution.sha256(),
        configuration_hash=CONFIG_HASH,
        seed=42,
        benchmark="mock",
        agent="test_agent",
        task_id=execution.task_id,
        run_index=0,
        success=True,
        score=1.0,
        evaluated_at=TIMESTAMP,
    )
    params.update(kwargs)
    return EvaluationRecord(**params)


def _make_metric(**kwargs) -> MetricRecord:
    params = dict(
        benchmark="mock",
        agent="test_agent",
        evaluation_count=1,
        success_rate=1.0,
        repeated_run_consistency=1.0,
        composite_reliability=1.0,
        computed_at=TIMESTAMP,
    )
    params.update(kwargs)
    return MetricRecord(**params)


def _make_ranking(**kwargs) -> RankingRecord:
    params = dict(
        benchmark="mock",
        ranking_type="success",
        rankings=(("agent-a", 0.9),),
        rank_map={"agent-a": 1},
        computed_at=TIMESTAMP,
    )
    params.update(kwargs)
    return RankingRecord(**params)


class TestEmptyExports:
    def test_export_empty_executions(self, tmp_path: Path):
        path = RecordExporter.export_executions([], tmp_path / "empty_execs.csv")
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 0
        assert reader.fieldnames is not None

    def test_export_empty_evaluations(self, tmp_path: Path):
        path = RecordExporter.export_evaluations([], tmp_path / "empty_evals.csv")
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 0

    def test_export_empty_metrics(self, tmp_path: Path):
        path = RecordExporter.export_metrics([], tmp_path / "empty_metrics.csv")
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 0

    def test_export_empty_rankings(self, tmp_path: Path):
        path = RecordExporter.export_rankings([], tmp_path / "empty_rankings.csv")
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 0


class TestOverwriteAndPaths:
    def test_export_overwrite_existing(self, tmp_path: Path):
        dest = tmp_path / "overwrite.csv"
        RecordExporter.export_executions([_make_execution("t1")], dest)
        RecordExporter.export_executions([_make_execution("t2")], dest)
        with dest.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["task_id"] == "t2"

    def test_export_suffix_handling(self, tmp_path: Path):
        records = [_make_execution("t1")]
        path = RecordExporter.export_executions(records, tmp_path / "data.txt")
        assert path.suffix == ".csv"
        assert path.exists()

        path2 = RecordExporter.export_evaluations(
            [_make_evaluation(_make_execution("t1"))], tmp_path / "evals.txt"
        )
        assert path2.suffix == ".csv"

    def test_export_all_creates_multiple_files(self, tmp_path: Path):
        execs = [_make_execution("t1")]
        evals = [_make_evaluation(execs[0])]
        metrics = [_make_metric()]
        rankings = [_make_ranking()]
        paths = RecordExporter.export_all(
            executions=execs,
            evaluations=evals,
            metrics=metrics,
            rankings=rankings,
            output_dir=str(tmp_path / "multi_export"),
        )
        assert set(paths.keys()) == {"executions", "evaluations", "metrics", "rankings"}
        for name in paths:
            assert paths[name].exists()

    def test_export_all_with_partial_data(self, tmp_path: Path):
        paths = RecordExporter.export_all(
            evaluations=[_make_evaluation(_make_execution("t1"))],
            output_dir=str(tmp_path / "partial"),
        )
        assert "evaluations" in paths
        assert "executions" not in paths
        assert "metrics" not in paths
        assert "rankings" not in paths

    def test_export_invalid_path_parent(self, tmp_path: Path):
        parent = tmp_path / "existing_file.csv"
        parent.write_text("not a directory")
        bad_path = parent / "sub" / "out.csv"
        with pytest.raises(OSError):
            RecordExporter.export_executions([_make_execution("t1")], bad_path)


class TestUnicode:
    def test_export_executions_with_unicode(self, tmp_path: Path):
        records = [
            _make_execution("t1", stdout="héllo wörld 🌍", stderr="café"),
            _make_execution("t2", stdout="中文测试", stderr="日本語"),
        ]
        path = RecordExporter.export_executions(records, tmp_path / "unicode_execs.csv")
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["stdout"] == "héllo wörld 🌍"
        assert rows[1]["stdout"] == "中文测试"

    def test_export_metrics_with_unicode(self, tmp_path: Path):
        records = [_make_metric(benchmark="测试", agent="代理")]
        path = RecordExporter.export_metrics(records, tmp_path / "unicode_metrics.csv")
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["benchmark"] == "测试"
        assert rows[0]["agent"] == "代理"


class TestErrorHandling:
    def test_export_executions_invalid_type(self, tmp_path: Path):
        with pytest.raises(TypeError):
            RecordExporter.export_executions(42, tmp_path / "bad.csv")
        with pytest.raises((TypeError, AttributeError)):
            RecordExporter.export_executions("not_a_list", tmp_path / "bad2.csv")

    def test_export_evaluations_invalid_type(self, tmp_path: Path):
        with pytest.raises(TypeError):
            RecordExporter.export_evaluations(None, tmp_path / "bad.csv")

    def test_export_metrics_invalid_type(self, tmp_path: Path):
        with pytest.raises(TypeError):
            RecordExporter.export_metrics(3.14, tmp_path / "bad.csv")


class TestLargeData:
    def test_export_large_dataset(self, tmp_path: Path):
        records = [_make_execution(f"task-{i}") for i in range(1000)]
        path = RecordExporter.export_executions(records, tmp_path / "large.csv")
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1000
        assert rows[0]["task_id"] == "task-0"
        assert rows[999]["task_id"] == "task-999"

    def test_export_large_evaluations(self, tmp_path: Path):
        execs = [_make_execution(f"task-{i}") for i in range(500)]
        evals = [_make_evaluation(e) for e in execs]
        path = RecordExporter.export_evaluations(evals, tmp_path / "large_evals.csv")
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 500
