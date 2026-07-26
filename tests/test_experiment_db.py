"""Tests for ExperimentDatabase (SQLite experiment repository)."""

import json
import tempfile
from pathlib import Path

import pytest

from llm_reliability.configs.config import Configuration
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord
from llm_reliability.repositories.experiment_db import ExperimentDatabase


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.db"
        edb = ExperimentDatabase(path)
        edb.connect()
        yield edb
        edb.close()


@pytest.fixture
def exec_record():
    return ExecutionRecord(
        configuration_hash="a" * 64,
        seed=42,
        benchmark="GAIA",
        agent="test_agent",
        task_id="t1",
        run_index=0,
        runtime_seconds=1.5,
        timestamp="2026-01-01T00:00:00+00:00",
        stdout="ok",
        stderr="",
        status="success",
    )


@pytest.fixture
def eval_record(exec_record):
    return EvaluationRecord.from_execution(
        execution=exec_record,
        success=True,
        score=1.0,
        metrics={"difficulty": "1"},
        evaluated_at="2026-01-01T01:00:00+00:00",
    )


@pytest.fixture
def metric_record():
    return MetricRecord(
        benchmark="GAIA",
        agent="test_agent",
        task_id=None,
        evaluation_count=1,
        success_rate=1.0,
        repeated_run_consistency=1.0,
        perturbation_robustness=None,
        fault_tolerance=None,
        isr_output=None,
        isr_behavior=None,
        isr_composite_val=None,
        composite_reliability=1.0,
        computed_at="2026-01-01T02:00:00+00:00",
    )


# ------------------------------------------------------------------
# Connection lifecycle
# ------------------------------------------------------------------


def test_connect_and_close():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lifecycle.db"
        edb = ExperimentDatabase(path)
        assert not edb.connected
        edb.connect()
        assert edb.connected
        edb.close()
        assert not edb.connected


def test_operation_without_connect_raises():
    edb = ExperimentDatabase(":memory:")
    with pytest.raises(RuntimeError, match="not connected"):
        edb.list_experiments()


# ------------------------------------------------------------------
# Experiment CRUD
# ------------------------------------------------------------------


def test_save_and_list_experiments(db):
    db.save_experiment("exp1", "Test Experiment 1")
    db.save_experiment("exp2", "Test Experiment 2")
    exps = db.list_experiments()
    assert len(exps) == 2
    names = {e["name"] for e in exps}
    assert names == {"Test Experiment 1", "Test Experiment 2"}


def test_save_and_get_experiment(db):
    config = Configuration(
        experiment_name="test",
        benchmark="test",
        agent="test",
        llm="test",
        prompt_version="1",
        dataset_version="1",
        seed=42,
        repetitions=1,
    )
    db.save_experiment("exp1", "My Experiment", config)
    exp = db.get_experiment("exp1")
    assert exp is not None
    assert exp["name"] == "My Experiment"
    parsed = json.loads(exp["config_json"])
    assert parsed["experiment_name"] == "test"


def test_update_experiment_status(db):
    db.save_experiment("exp1", "Test")
    db.update_experiment_status("exp1", "completed")
    exp = db.get_experiment("exp1")
    assert exp["status"] == "completed"


def test_get_nonexistent_experiment(db):
    exp = db.get_experiment("nonexistent")
    assert exp is None


# ------------------------------------------------------------------
# Execution persistence
# ------------------------------------------------------------------


def test_save_and_query_executions(db, exec_record):
    db.save_experiment("exp1", "Test")
    db.save_execution("exp1", exec_record)
    results = db.query_executions(experiment_id="exp1")
    assert len(results) == 1
    assert results[0]["task_id"] == "t1"
    assert results[0]["benchmark"] == "GAIA"


def test_save_executions_bulk(db, exec_record):
    db.save_experiment("exp1", "Test")
    records = [
        ExecutionRecord(
            configuration_hash="a" * 64,
            seed=42,
            benchmark="GAIA",
            agent="test_agent",
            task_id=f"t{i}",
            run_index=0,
            runtime_seconds=float(i),
            timestamp="2026-01-01T00:00:00+00:00",
            stdout="ok",
            stderr="",
            status="success",
        )
        for i in range(3)
    ]
    db.save_executions("exp1", records)
    results = db.query_executions(experiment_id="exp1")
    assert len(results) == 3


def test_query_executions_by_benchmark(db, exec_record):
    db.save_experiment("exp1", "Test")
    db.save_execution("exp1", exec_record)
    results = db.query_executions(benchmark="GAIA")
    assert len(results) == 1
    results = db.query_executions(benchmark="MMLU")
    assert len(results) == 0


def test_query_executions_by_status(db, exec_record):
    db.save_experiment("exp1", "Test")
    db.save_execution("exp1", exec_record)
    results = db.query_executions(status="success")
    assert len(results) == 1
    results = db.query_executions(status="error")
    assert len(results) == 0


# ------------------------------------------------------------------
# Evaluation persistence
# ------------------------------------------------------------------


def test_save_and_query_evaluations(db, exec_record, eval_record):
    db.save_experiment("exp1", "Test")
    db.save_execution("exp1", exec_record)
    db.save_evaluation("exp1", eval_record)
    results = db.query_evaluations(experiment_id="exp1")
    assert len(results) == 1
    assert results[0]["success"] == 1
    assert results[0]["score"] == 1.0


def test_query_evaluations_by_success(db, eval_record):
    db.save_experiment("exp1", "Test")
    exec_r = ExecutionRecord(
        configuration_hash="a" * 64,
        seed=42,
        benchmark="GAIA",
        agent="test_agent",
        task_id="t1",
        run_index=0,
        runtime_seconds=1.0,
        timestamp="2026-01-01T00:00:00+00:00",
        stdout="ok",
        stderr="",
        status="success",
    )
    db.save_execution("exp1", exec_r)
    db.save_evaluation("exp1", eval_record)
    results = db.query_evaluations(successful=True)
    assert len(results) == 1
    results = db.query_evaluations(successful=False)
    assert len(results) == 0


# ------------------------------------------------------------------
# Metric persistence
# ------------------------------------------------------------------


def test_save_and_query_metrics(db, metric_record):
    db.save_experiment("exp1", "Test")
    db.save_metric("exp1", metric_record)
    results = db.query_metrics(experiment_id="exp1")
    assert len(results) == 1
    assert results[0]["composite_reliability"] == 1.0


def test_query_metrics_by_benchmark(db, metric_record):
    db.save_experiment("exp1", "Test")
    db.save_metric("exp1", metric_record)
    results = db.query_metrics(benchmark="GAIA")
    assert len(results) == 1
    results = db.query_metrics(benchmark="MMLU")
    assert len(results) == 0


# ------------------------------------------------------------------
# Cross-experiment comparison
# ------------------------------------------------------------------


def test_compare_agents_across_experiments(db, metric_record):
    db.save_experiment("exp1", "Experiment 1")
    db.save_experiment("exp2", "Experiment 2")
    db.save_metric("exp1", metric_record)
    m2 = MetricRecord(
        benchmark="GAIA",
        agent="other_agent",
        task_id=None,
        evaluation_count=5,
        success_rate=0.8,
        repeated_run_consistency=0.7,
        perturbation_robustness=None,
        fault_tolerance=None,
        isr_output=None,
        isr_behavior=None,
        isr_composite_val=None,
        composite_reliability=0.75,
        computed_at="2026-01-01T03:00:00+00:00",
    )
    db.save_metric("exp2", m2)
    results = db.compare_agents_across_experiments(["exp1", "exp2"])
    assert len(results) == 2
    assert results[0]["experiment_id"] in ("exp1", "exp2")


def test_compare_empty_experiments(db):
    results = db.compare_agents_across_experiments([])
    assert results == []


# ------------------------------------------------------------------
# Ranking
# ------------------------------------------------------------------


def test_get_ranking(db, metric_record):
    db.save_experiment("exp1", "Test")
    db.save_metric("exp1", metric_record)
    m2 = MetricRecord(
        benchmark="GAIA",
        agent="agent_b",
        task_id=None,
        evaluation_count=1,
        success_rate=0.5,
        repeated_run_consistency=0.5,
        perturbation_robustness=None,
        fault_tolerance=None,
        isr_output=None,
        isr_behavior=None,
        isr_composite_val=None,
        composite_reliability=0.5,
        computed_at="2026-01-01T03:00:00+00:00",
    )
    db.save_metric("exp1", m2)
    ranking = db.get_ranking("exp1", benchmark="GAIA")
    assert len(ranking) == 2
    assert ranking[0]["agent"] == "test_agent"  # ranked by composite_reliability desc
    assert ranking[0]["composite_reliability"] == 1.0


def test_get_ranking_without_benchmark_filter(db, metric_record):
    db.save_experiment("exp1", "Test")
    db.save_metric("exp1", metric_record)
    ranking = db.get_ranking("exp1")
    assert len(ranking) == 1


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------


def test_duplicate_experiment_id_is_replaced(db):
    db.save_experiment("exp1", "Original")
    db.save_experiment("exp1", "Replacement")
    exp = db.get_experiment("exp1")
    assert exp["name"] == "Replacement"


def test_save_evaluation_without_experiment(db, eval_record):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        db.save_evaluation("nonexistent", eval_record)
