"""Tests for EvaluationRecord (Artifact 5)."""

import pytest
from pydantic import ValidationError

from llm_reliability.records import EvaluationRecord, ExecutionRecord
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


def test_from_execution_links_hashes() -> None:
    execution = _make_execution()
    evaluation = EvaluationRecord.from_execution(
        execution,
        success=True,
        score=1.0,
        evaluated_at=TIMESTAMP,
    )
    assert evaluation.execution_hash == execution.sha256()
    assert evaluation.configuration_hash == execution.configuration_hash


def test_from_execution_propagates_identity_fields() -> None:
    execution = _make_execution(
        seed=7,
        benchmark="gaia",
        agent="agent_x",
        task_id="task-99",
        run_index=3,
        perturbation="typo",
        fault_injected=True,
    )
    evaluation = EvaluationRecord.from_execution(
        execution,
        success=True,
        score=0.75,
        evaluated_at=TIMESTAMP,
    )
    assert evaluation.seed == 7
    assert evaluation.benchmark == "gaia"
    assert evaluation.agent == "agent_x"
    assert evaluation.task_id == "task-99"
    assert evaluation.run_index == 3
    assert evaluation.perturbation == "typo"
    assert evaluation.fault_injected is True


def test_from_execution_stores_benchmark_metrics() -> None:
    execution = _make_execution()
    evaluation = EvaluationRecord.from_execution(
        execution,
        success=False,
        score=0.0,
        metrics={"detail": "wrong answer", "partial": 0.0},
        evaluated_at=TIMESTAMP,
    )
    assert evaluation.success is False
    assert evaluation.metrics["detail"] == "wrong answer"


def test_serialization_round_trip() -> None:
    execution = _make_execution()
    evaluation = EvaluationRecord.from_execution(
        execution,
        success=True,
        score=0.5,
        evaluated_at=TIMESTAMP,
    )
    restored = EvaluationRecord.from_canonical_json(evaluation.canonical_json())
    assert evaluation == restored
    assert evaluation.sha256() == restored.sha256()


def test_hash_is_deterministic() -> None:
    execution = _make_execution()
    first = EvaluationRecord.from_execution(
        execution,
        success=True,
        score=1.0,
        evaluated_at=TIMESTAMP,
    )
    second = EvaluationRecord.from_execution(
        execution,
        success=True,
        score=1.0,
        evaluated_at=TIMESTAMP,
    )
    assert first.sha256() == second.sha256()


def test_rejects_score_above_one() -> None:
    execution = _make_execution()
    with pytest.raises(ValidationError):
        EvaluationRecord.from_execution(
            execution,
            success=True,
            score=1.5,
            evaluated_at=TIMESTAMP,
        )


def test_rejects_negative_score() -> None:
    execution = _make_execution()
    with pytest.raises(ValidationError):
        EvaluationRecord.from_execution(
            execution,
            success=True,
            score=-0.1,
            evaluated_at=TIMESTAMP,
        )


def test_rejects_invalid_execution_hash_length() -> None:
    with pytest.raises(ValidationError):
        EvaluationRecord(
            execution_hash="short",
            configuration_hash=CONFIG_HASH,
            seed=42,
            benchmark="mock",
            agent="mock_agent",
            task_id="task-1",
            run_index=0,
            success=True,
            score=1.0,
            evaluated_at=TIMESTAMP,
        )


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        EvaluationRecord(
            execution_hash=CONFIG_HASH,
            configuration_hash=CONFIG_HASH,
            seed=42,
            benchmark="mock",
            agent="mock_agent",
            task_id="task-1",
            run_index=0,
            success=True,
            score=1.0,
            evaluated_at=TIMESTAMP,
            extra=True,
        )


def test_immutable() -> None:
    execution = _make_execution()
    evaluation = EvaluationRecord.from_execution(
        execution,
        success=True,
        score=1.0,
        evaluated_at=TIMESTAMP,
    )
    with pytest.raises(ValidationError):
        evaluation.success = False
