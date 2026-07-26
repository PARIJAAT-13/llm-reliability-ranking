"""Tests for MetricRecord (Artifact 6)."""

import pytest
from pydantic import ValidationError

from llm_reliability.records import EvaluationRecord, ExecutionRecord, MetricRecord
from tests.conftest import CONFIG_HASH, TIMESTAMP


def _make_execution(**overrides: object) -> ExecutionRecord:
    defaults = {
        "configuration_hash": CONFIG_HASH,
        "seed": 42,
        "benchmark": "mock",
        "agent": "mock_agent",
        "task_id": "task-1",
        "run_index": 0,
        "runtime_seconds": 1.0,
        "timestamp": TIMESTAMP,
        "stdout": "ok",
        "stderr": "",
        "status": "success",
    }
    defaults.update(overrides)
    return ExecutionRecord(**defaults)


def _make_evaluation(
    execution: ExecutionRecord,
    *,
    success: bool,
    score: float,
    metrics: dict[str, object] | None = None,
) -> EvaluationRecord:
    return EvaluationRecord.from_execution(
        execution,
        success=success,
        score=score,
        metrics=metrics,
        evaluated_at=TIMESTAMP,
    )


def test_from_evaluations_computes_success_rate() -> None:
    evaluations = [
        _make_evaluation(_make_execution(run_index=0), success=True, score=1.0),
        _make_evaluation(_make_execution(run_index=1), success=False, score=0.0),
    ]
    metric = MetricRecord.from_evaluations(evaluations, computed_at=TIMESTAMP)
    assert metric.success_rate == 0.5
    assert metric.evaluation_count == 2


def test_from_evaluations_requires_non_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MetricRecord.from_evaluations([], computed_at=TIMESTAMP)


def test_from_evaluations_rejects_mixed_agents() -> None:
    evaluations = [
        _make_evaluation(_make_execution(agent="agent_a"), success=True, score=1.0),
        _make_evaluation(_make_execution(agent="agent_b"), success=True, score=1.0),
    ]
    with pytest.raises(ValueError, match="same benchmark and agent"):
        MetricRecord.from_evaluations(evaluations, computed_at=TIMESTAMP)


def test_fault_tolerance_from_execution_context() -> None:
    evaluations = [
        _make_evaluation(
            _make_execution(run_index=0, fault_injected=True),
            success=True,
            score=1.0,
        ),
        _make_evaluation(
            _make_execution(run_index=1, fault_injected=True),
            success=False,
            score=0.0,
        ),
    ]
    metric = MetricRecord.from_evaluations(evaluations, computed_at=TIMESTAMP)
    assert metric.fault_tolerance == 0.5


def test_serialization_round_trip() -> None:
    evaluations = [
        _make_evaluation(_make_execution(), success=True, score=1.0),
    ]
    metric = MetricRecord.from_evaluations(evaluations, computed_at=TIMESTAMP)
    restored = MetricRecord.from_canonical_json(metric.canonical_json())
    assert metric == restored


def test_immutable() -> None:
    evaluations = [
        _make_evaluation(_make_execution(), success=True, score=1.0),
    ]
    metric = MetricRecord.from_evaluations(evaluations, computed_at=TIMESTAMP)
    with pytest.raises(ValidationError):
        metric.success_rate = 0.0
