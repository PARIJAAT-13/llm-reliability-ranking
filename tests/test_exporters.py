"""Tests for RecordExporter."""

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


def _make_execution(task_id: str = "task-0") -> ExecutionRecord:
    return ExecutionRecord(
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


def _make_evaluation(execution: ExecutionRecord) -> EvaluationRecord:
    return EvaluationRecord(
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


def test_export_executions_csv(tmp_path: Path):
    records = [_make_execution("t1"), _make_execution("t2")]
    path = RecordExporter.export_executions(records, tmp_path / "execs.csv")
    assert path.exists()
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["task_id"] == "t1"
    assert rows[1]["task_id"] == "t2"


def test_export_evaluations_csv(tmp_path: Path):
    execs = [_make_execution("t1"), _make_execution("t2")]
    evals = [_make_evaluation(e) for e in execs]
    path = RecordExporter.export_evaluations(evals, tmp_path / "evals.csv")
    assert path.exists()
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_export_metrics_csv(tmp_path: Path):
    records = [
        MetricRecord(
            benchmark="mock",
            agent="test_agent",
            evaluation_count=1,
            success_rate=1.0,
            repeated_run_consistency=1.0,
            composite_reliability=1.0,
            computed_at=TIMESTAMP,
        )
    ]
    path = RecordExporter.export_metrics(records, tmp_path / "metrics.csv")
    assert path.exists()
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1


def test_export_rankings_csv(tmp_path: Path):
    records = [
        RankingRecord(
            benchmark="mock",
            ranking_type="success",
            rankings=(("agent-a", 0.9),),
            rank_map={"agent-a": 1},
            computed_at=TIMESTAMP,
        )
    ]
    path = RecordExporter.export_rankings(records, tmp_path / "rankings.csv")
    assert path.exists()
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1


def test_export_all(tmp_path: Path):
    execs = [_make_execution("t1")]
    evals = [_make_evaluation(execs[0])]
    paths = RecordExporter.export_all(
        executions=execs,
        evaluations=evals,
        output_dir=str(tmp_path / "exports"),
    )
    assert "executions" in paths
    assert "evaluations" in paths
    assert paths["executions"].exists()
    assert paths["evaluations"].exists()


def test_csv_has_header(tmp_path: Path):
    records = [_make_execution()]
    path = RecordExporter.export_executions(records, tmp_path / "out.csv")
    with path.open(encoding="utf-8") as f:
        first_line = f.readline().strip()
    assert "task_id" in first_line
    assert "status" in first_line
