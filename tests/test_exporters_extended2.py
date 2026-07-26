"""Extended tests for RecordExporter — 40+ tests."""

from __future__ import annotations

import csv
import json
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


class TestExportExecutions:
    def test_single_execution(self, tmp_path: Path):
        rec = _make_execution()
        path = RecordExporter.export_executions([rec], tmp_path / "execs.csv")
        assert path.exists()
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["task_id"] == "task-0"

    def test_multiple_executions(self, tmp_path: Path):
        recs = [_make_execution(task_id=f"task-{i}") for i in range(10)]
        path = RecordExporter.export_executions(recs, tmp_path / "execs.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 10

    def test_execution_columns_present(self, tmp_path: Path):
        rec = _make_execution()
        path = RecordExporter.export_executions([rec], tmp_path / "execs.csv")
        with path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert "task_id" in rows[0]
        assert "status" in rows[0]
        assert "runtime_seconds" in rows[0]

    def test_execution_with_error(self, tmp_path: Path):
        rec = _make_execution(status="error", error="timeout")
        path = RecordExporter.export_executions([rec], tmp_path / "execs.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["status"] == "error"
        assert rows[0]["error"] == "timeout"


class TestExportEvaluations:
    def test_single_evaluation(self, tmp_path: Path):
        ex = _make_execution()
        ev = _make_evaluation(ex)
        path = RecordExporter.export_evaluations([ev], tmp_path / "evals.csv")
        assert path.exists()
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_evaluation_scores(self, tmp_path: Path):
        ex = _make_execution()
        evals = [
            _make_evaluation(ex, score=0.5, success=False),
            _make_evaluation(ex, score=1.0, success=True),
        ]
        path = RecordExporter.export_evaluations(evals, tmp_path / "evals.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        scores = [float(r["score"]) for r in rows]
        assert 0.5 in scores
        assert 1.0 in scores


class TestExportMetrics:
    def test_single_metric(self, tmp_path: Path):
        m = _make_metric()
        path = RecordExporter.export_metrics([m], tmp_path / "metrics.csv")
        assert path.exists()
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_metric_values(self, tmp_path: Path):
        m = _make_metric(success_rate=0.85, composite_reliability=0.75)
        path = RecordExporter.export_metrics([m], tmp_path / "metrics.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["success_rate"]) == 0.85
        assert float(rows[0]["composite_reliability"]) == 0.75


class TestExportRankings:
    def test_single_ranking(self, tmp_path: Path):
        r = _make_ranking()
        path = RecordExporter.export_rankings([r], tmp_path / "rankings.csv")
        assert path.exists()
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_ranking_multiple_agents(self, tmp_path: Path):
        r = _make_ranking(
            rankings=(("a", 0.9), ("b", 0.8)),
            rank_map={"a": 1, "b": 2},
        )
        path = RecordExporter.export_rankings([r], tmp_path / "rankings.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1


class TestExportAll:
    def test_export_all_types(self, tmp_path: Path):
        ex = _make_execution()
        ev = _make_evaluation(ex)
        m = _make_metric()
        r = _make_ranking()
        result = RecordExporter.export_all(
            executions=[ex],
            evaluations=[ev],
            metrics=[m],
            rankings=[r],
            output_dir=str(tmp_path / "all"),
        )
        assert "executions" in result
        assert "evaluations" in result
        assert "metrics" in result
        assert "rankings" in result
        for p in result.values():
            assert Path(p).exists()

    def test_export_all_partial(self, tmp_path: Path):
        ex = _make_execution()
        result = RecordExporter.export_all(
            executions=[ex],
            output_dir=str(tmp_path / "partial"),
        )
        assert "executions" in result
        assert "evaluations" not in result
        assert "metrics" not in result
        assert "rankings" not in result

    def test_export_all_empty(self, tmp_path: Path):
        result = RecordExporter.export_all(output_dir=str(tmp_path / "empty"))
        assert result == {}

    def test_export_all_creates_directory(self, tmp_path: Path):
        result = RecordExporter.export_all(
            executions=[_make_execution()],
            output_dir=str(tmp_path / "new_dir" / "sub"),
        )
        assert Path(result["executions"]).exists()


class TestExportEdgeCases:
    def test_unicode_in_fields(self, tmp_path: Path):
        rec = _make_execution(stdout="héllo wörld 🌍", task_id="unicode-τâsk")
        path = RecordExporter.export_executions([rec], tmp_path / "unicode.csv")
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["stdout"] == "héllo wörld 🌍"
        assert rows[0]["task_id"] == "unicode-τâsk"

    def test_special_chars_in_fields(self, tmp_path: Path):
        rec = _make_execution(stdout='line1\nline2,"quote",spaces')
        path = RecordExporter.export_executions([rec], tmp_path / "special.csv")
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["stdout"] == 'line1\nline2,"quote",spaces'

    def test_large_number_of_records(self, tmp_path: Path):
        recs = [_make_execution(task_id=f"task-{i}") for i in range(1000)]
        path = RecordExporter.export_executions(recs, tmp_path / "large.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1000

    def test_records_with_none_fields(self, tmp_path: Path):
        rec = _make_execution(error=None, perturbation=None)
        path = RecordExporter.export_executions([rec], tmp_path / "none.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_path_without_csv_suffix(self, tmp_path: Path):
        rec = _make_execution()
        path = RecordExporter.export_executions([rec], tmp_path / "noext")
        assert path.suffix == ".csv"
        assert path.exists()

    def test_auto_csv_suffix_added(self, tmp_path: Path):
        rec = _make_execution()
        path = RecordExporter.export_executions([rec], tmp_path / "data" / "output")
        assert path.suffix == ".csv"
        assert path.exists()

    def test_empty_list_creates_file_with_headers(self, tmp_path: Path):
        path = RecordExporter.export_executions([], tmp_path / "empty.csv")
        assert path.exists()
        with path.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert len(list(reader)) == 0

    def test_empty_evaluations(self, tmp_path: Path):
        path = RecordExporter.export_evaluations([], tmp_path / "empty_evals.csv")
        assert path.exists()
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 0

    def test_empty_metrics(self, tmp_path: Path):
        path = RecordExporter.export_metrics([], tmp_path / "empty_metrics.csv")
        assert path.exists()
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 0

    def test_empty_rankings(self, tmp_path: Path):
        path = RecordExporter.export_rankings([], tmp_path / "empty_rankings.csv")
        assert path.exists()
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 0

    def test_evaluation_with_metrics_field(self, tmp_path: Path):
        ex = _make_execution()
        ev = _make_evaluation(ex, metrics={"accuracy": 0.95, "f1": 0.89})
        path = RecordExporter.export_evaluations([ev], tmp_path / "metrics_field.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_execution_with_software_versions(self, tmp_path: Path):
        rec = _make_execution(software_versions={"python": "3.11", "torch": "2.1.0"})
        path = RecordExporter.export_executions([rec], tmp_path / "sw.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_ranking_with_multiple_entries(self, tmp_path: Path):
        r = _make_ranking(
            rankings=(("a", 0.9), ("b", 0.8), ("c", 0.7)),
            rank_map={"a": 1, "b": 2, "c": 3},
        )
        path = RecordExporter.export_rankings([r], tmp_path / "multi_rank.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_execution_with_environment_metadata(self, tmp_path: Path):
        rec = _make_execution(environment_metadata={"gpu": "A100", "cluster": "prod"})
        path = RecordExporter.export_executions([rec], tmp_path / "env_meta.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_execution_with_agent_output(self, tmp_path: Path):
        rec = _make_execution(agent_output={"result": "passed", "score": 95})
        path = RecordExporter.export_executions([rec], tmp_path / "agent_out.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_execution_various_statuses(self, tmp_path: Path):
        for status in ("success", "failure", "error", "timeout"):
            rec = _make_execution(task_id=f"task-{status}", status=status)
            path = RecordExporter.export_executions([rec], tmp_path / f"{status}.csv")
            with path.open() as f:
                rows = list(csv.DictReader(f))
            assert rows[0]["status"] == status

    def test_evaluation_various_scores(self, tmp_path: Path):
        ex = _make_execution()
        evals = [_make_evaluation(ex, score=i / 10, success=i >= 5) for i in range(11)]
        path = RecordExporter.export_evaluations(evals, tmp_path / "scores.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 11

    def test_metrics_with_perturbation_fields(self, tmp_path: Path):
        m = _make_metric(
            perturbation_robustness=0.85,
            fault_tolerance=0.75,
            repeated_run_consistency=0.9,
        )
        path = RecordExporter.export_metrics([m], tmp_path / "perturb.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["perturbation_robustness"]) == 0.85
        assert float(rows[0]["fault_tolerance"]) == 0.75

    def test_metrics_with_isr_fields(self, tmp_path: Path):
        m = _make_metric(isr_output=0.9, isr_behavior=0.8, isr_composite_val=0.85)
        path = RecordExporter.export_metrics([m], tmp_path / "isr.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["isr_output"]) == 0.9
        assert float(rows[0]["isr_behavior"]) == 0.8

    def test_ranking_reliability_type(self, tmp_path: Path):
        r = _make_ranking(ranking_type="reliability")
        path = RecordExporter.export_rankings([r], tmp_path / "rel_rank.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["ranking_type"] == "reliability"

    def test_export_all_mixed_types(self, tmp_path: Path):
        ex = _make_execution()
        ev = _make_evaluation(ex)
        result = RecordExporter.export_all(
            executions=[ex],
            evaluations=[ev],
            output_dir=str(tmp_path / "mixed"),
        )
        assert len(result) == 2

    def test_output_path_created_with_parent_dirs(self, tmp_path: Path):
        rec = _make_execution()
        path = RecordExporter.export_executions([rec], tmp_path / "a" / "b" / "c" / "deep.csv")
        assert path.exists()
        assert path.parent.name == "c"

    def test_multiple_metrics_export(self, tmp_path: Path):
        ms = [_make_metric(agent=f"agent-{i}", success_rate=(i + 1) / 10) for i in range(5)]
        path = RecordExporter.export_metrics(ms, tmp_path / "multi_metrics.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5
        rates = sorted(float(r["success_rate"]) for r in rows)
        assert rates == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_evaluation_with_different_agents(self, tmp_path: Path):
        ex = _make_execution()
        evals = [
            _make_evaluation(ex, agent="a", task_id="t1"),
            _make_evaluation(ex, agent="b", task_id="t2"),
        ]
        path = RecordExporter.export_evaluations(evals, tmp_path / "multi_agents.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert {r["agent"] for r in rows} == {"a", "b"}
