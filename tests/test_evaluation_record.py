"""Tests for the new EvaluationRecord model (Artifact 5)."""

import json
import pytest
from pydantic import ValidationError
from records.evaluation_record import EvaluationRecord


def test_evaluation_record_instantiation() -> None:
    """Test that EvaluationRecord can be successfully instantiated with valid fields."""
    record = EvaluationRecord(
        execution_id="exec-123",
        benchmark="gsm8k",
        task_id="task-456",
        success=True,
        score=4.0,
        max_score=5.0,
        passed=True,
        evaluation_time_seconds=1.5,
        evaluator_version="1.0.0",
        metadata={"detail": "all tests passed"}
    )
    assert record.execution_id == "exec-123"
    assert record.benchmark == "gsm8k"
    assert record.task_id == "task-456"
    assert record.success is True
    assert record.score == 4.0
    assert record.max_score == 5.0
    assert record.passed is True
    assert record.evaluation_time_seconds == 1.5
    assert record.evaluator_version == "1.0.0"
    assert record.metadata == {"detail": "all tests passed"}


def test_evaluation_record_immutability() -> None:
    """Test that EvaluationRecord is immutable and fields cannot be modified after creation."""
    record = EvaluationRecord(
        execution_id="exec-123",
        benchmark="gsm8k",
        task_id="task-456",
        success=True,
        score=4.0,
        max_score=5.0,
        passed=True,
        evaluation_time_seconds=1.5,
        evaluator_version="1.0.0",
        metadata={}
    )
    with pytest.raises(ValidationError):
        record.success = False  # type: ignore[misc]

    with pytest.raises(ValidationError):
        record.score = 5.0  # type: ignore[misc]


def test_evaluation_record_rejects_unknown_fields() -> None:
    """Test that EvaluationRecord rejects unknown fields at instantiation."""
    with pytest.raises(ValidationError):
        EvaluationRecord(
            execution_id="exec-123",
            benchmark="gsm8k",
            task_id="task-456",
            success=True,
            score=4.0,
            max_score=5.0,
            passed=True,
            evaluation_time_seconds=1.5,
            evaluator_version="1.0.0",
            extra_field="invalid",  # type: ignore[call-arg]
        )


def test_evaluation_record_validation_failures() -> None:
    """Test that invalid values for numeric fields raise ValidationError."""
    # score >= 0
    with pytest.raises(ValidationError):
        EvaluationRecord(
            execution_id="exec-123",
            benchmark="gsm8k",
            task_id="task-456",
            success=True,
            score=-1.0,
            max_score=5.0,
            passed=True,
            evaluation_time_seconds=1.5,
            evaluator_version="1.0.0",
        )

    # max_score > 0
    with pytest.raises(ValidationError):
        EvaluationRecord(
            execution_id="exec-123",
            benchmark="gsm8k",
            task_id="task-456",
            success=True,
            score=4.0,
            max_score=0.0,
            passed=True,
            evaluation_time_seconds=1.5,
            evaluator_version="1.0.0",
        )

    with pytest.raises(ValidationError):
        EvaluationRecord(
            execution_id="exec-123",
            benchmark="gsm8k",
            task_id="task-456",
            success=True,
            score=4.0,
            max_score=-2.0,
            passed=True,
            evaluation_time_seconds=1.5,
            evaluator_version="1.0.0",
        )

    # evaluation_time_seconds >= 0
    with pytest.raises(ValidationError):
        EvaluationRecord(
            execution_id="exec-123",
            benchmark="gsm8k",
            task_id="task-456",
            success=True,
            score=4.0,
            max_score=5.0,
            passed=True,
            evaluation_time_seconds=-0.5,
            evaluator_version="1.0.0",
        )


def test_evaluation_record_deterministic_hash() -> None:
    """Test that the SHA-256 hash is deterministic and matches for identical data."""
    record1 = EvaluationRecord(
        execution_id="exec-123",
        benchmark="gsm8k",
        task_id="task-456",
        success=True,
        score=4.0,
        max_score=5.0,
        passed=True,
        evaluation_time_seconds=1.5,
        evaluator_version="1.0.0",
        metadata={"a": 1, "b": 2}
    )
    record2 = EvaluationRecord(
        execution_id="exec-123",
        benchmark="gsm8k",
        task_id="task-456",
        success=True,
        score=4.0,
        max_score=5.0,
        passed=True,
        evaluation_time_seconds=1.5,
        evaluator_version="1.0.0",
        metadata={"b": 2, "a": 1}
    )
    assert record1.sha256() == record2.sha256()

    # Slight modification results in a different hash
    record3 = EvaluationRecord(
        execution_id="exec-123",
        benchmark="gsm8k",
        task_id="task-456",
        success=True,
        score=4.0,
        max_score=5.0,
        passed=True,
        evaluation_time_seconds=1.5,
        evaluator_version="1.0.1",
        metadata={"a": 1, "b": 2}
    )
    assert record1.sha256() != record3.sha256()


def test_evaluation_record_serialization_deserialization_round_trip() -> None:
    """Test that EvaluationRecord round trips through canonical JSON and dictionary."""
    record = EvaluationRecord(
        execution_id="exec-123",
        benchmark="gsm8k",
        task_id="task-456",
        success=True,
        score=4.0,
        max_score=5.0,
        passed=True,
        evaluation_time_seconds=1.5,
        evaluator_version="1.0.0",
        metadata={"detail": "some detail"}
    )
    
    # Test dictionary round trip
    dumped_dict = record.canonical_dict()
    assert isinstance(dumped_dict, dict)
    restored_from_dict = EvaluationRecord.model_validate(dumped_dict)
    assert record == restored_from_dict

    # Test JSON round trip
    json_str = record.canonical_json()
    assert isinstance(json_str, str)
    restored_from_json = EvaluationRecord.from_canonical_json(json_str)
    assert record == restored_from_json
    assert record.sha256() == restored_from_json.sha256()


def test_evaluation_record_canonical_json_properties() -> None:
    """Test properties of canonical JSON: sorted keys, no whitespace in separators."""
    record = EvaluationRecord(
        execution_id="exec-123",
        benchmark="gsm8k",
        task_id="task-456",
        success=True,
        score=4.0,
        max_score=5.0,
        passed=True,
        evaluation_time_seconds=1.5,
        evaluator_version="1.0.0",
        metadata={"z": 10, "a": 1}
    )
    json_str = record.canonical_json()
    
    # Compact format: no spaces between key-values or elements.
    assert " " not in json_str
    
    # Ensure keys are sorted alphabetically.
    parsed = json.loads(json_str)
    sorted_keys = sorted(list(parsed.keys()))
    key_positions = [json_str.find(f'"{k}"') for k in sorted_keys]
    assert key_positions == sorted(key_positions)
