"""Tests for the new ExecutionRecord model (Artifact 4)."""

import json
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from records.execution_record import ExecutionRecord


def test_execution_record_instantiation() -> None:
    """Test that ExecutionRecord can be successfully instantiated with valid fields."""
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
    record = ExecutionRecord(
        execution_id="exec-123",
        configuration_hash="a" * 64,
        benchmark="gsm8k",
        agent="agent-bob",
        task_id="task-456",
        status="SUCCESS",
        start_time=start,
        end_time=end,
        runtime_seconds=10.0,
        stdout="Job started\nJob finished",
        stderr="",
        error_message=None,
        raw_output={"result": 42},
        metadata={"run": 1}
    )
    assert record.execution_id == "exec-123"
    assert record.configuration_hash == "a" * 64
    assert record.benchmark == "gsm8k"
    assert record.agent == "agent-bob"
    assert record.task_id == "task-456"
    assert record.status == "SUCCESS"
    assert record.start_time == start
    assert record.end_time == end
    assert record.runtime_seconds == 10.0
    assert record.stdout == "Job started\nJob finished"
    assert record.stderr == ""
    assert record.error_message is None
    assert record.raw_output == {"result": 42}
    assert record.metadata == {"run": 1}


def test_execution_record_immutability() -> None:
    """Test that ExecutionRecord is immutable and fields cannot be modified after creation."""
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
    record = ExecutionRecord(
        execution_id="exec-123",
        configuration_hash="a" * 64,
        benchmark="gsm8k",
        agent="agent-bob",
        task_id="task-456",
        status="SUCCESS",
        start_time=start,
        end_time=end,
        runtime_seconds=10.0,
        stdout="Job started",
        stderr="",
        error_message=None,
        raw_output={},
        metadata={}
    )
    with pytest.raises(ValidationError):
        record.status = "FAILED"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        record.runtime_seconds = 12.0  # type: ignore[misc]


def test_execution_record_rejects_unknown_fields() -> None:
    """Test that ExecutionRecord rejects unknown fields at instantiation."""
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
    with pytest.raises(ValidationError):
        ExecutionRecord(
            execution_id="exec-123",
            configuration_hash="a" * 64,
            benchmark="gsm8k",
            agent="agent-bob",
            task_id="task-456",
            status="SUCCESS",
            start_time=start,
            end_time=end,
            runtime_seconds=10.0,
            stdout="",
            stderr="",
            error_message=None,
            raw_output={},
            metadata={},
            extra_field="forbidden"  # type: ignore[call-arg]
        )


def test_execution_record_invalid_status() -> None:
    """Test that status must be one of SUCCESS, FAILED, TIMEOUT, ERROR."""
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
    for invalid_status in ["success", "failed", "COMPLETED", "RUNNING", ""]:
        with pytest.raises(ValidationError):
            ExecutionRecord(
                execution_id="exec-123",
                configuration_hash="a" * 64,
                benchmark="gsm8k",
                agent="agent-bob",
                task_id="task-456",
                status=invalid_status,  # type: ignore[arg-type]
                start_time=start,
                end_time=end,
                runtime_seconds=10.0,
                stdout="",
                stderr="",
                error_message=None,
                raw_output={},
                metadata={}
            )


def test_execution_record_negative_runtime() -> None:
    """Test that negative runtime_seconds values raise ValidationError."""
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
    with pytest.raises(ValidationError):
        ExecutionRecord(
            execution_id="exec-123",
            configuration_hash="a" * 64,
            benchmark="gsm8k",
            agent="agent-bob",
            task_id="task-456",
            status="SUCCESS",
            start_time=start,
            end_time=end,
            runtime_seconds=-0.1,  # Invalid
            stdout="",
            stderr="",
            error_message=None,
            raw_output={},
            metadata={}
        )


def test_execution_record_deterministic_hash() -> None:
    """Test that the SHA-256 hash is deterministic and identical for identical data."""
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
    record1 = ExecutionRecord(
        execution_id="exec-123",
        configuration_hash="a" * 64,
        benchmark="gsm8k",
        agent="agent-bob",
        task_id="task-456",
        status="SUCCESS",
        start_time=start,
        end_time=end,
        runtime_seconds=10.0,
        stdout="",
        stderr="",
        error_message=None,
        raw_output={"x": 1, "y": 2},
        metadata={"tag": "test"}
    )
    record2 = ExecutionRecord(
        execution_id="exec-123",
        configuration_hash="a" * 64,
        benchmark="gsm8k",
        agent="agent-bob",
        task_id="task-456",
        status="SUCCESS",
        start_time=start,
        end_time=end,
        runtime_seconds=10.0,
        stdout="",
        stderr="",
        error_message=None,
        raw_output={"y": 2, "x": 1},
        metadata={"tag": "test"}
    )
    assert record1.sha256() == record2.sha256()

    record3 = ExecutionRecord(
        execution_id="exec-123",
        configuration_hash="a" * 64,
        benchmark="gsm8k",
        agent="agent-bob",
        task_id="task-456",
        status="SUCCESS",
        start_time=start,
        end_time=end,
        runtime_seconds=10.1,
        stdout="",
        stderr="",
        error_message=None,
        raw_output={"x": 1, "y": 2},
        metadata={"tag": "test"}
    )
    assert record1.sha256() != record3.sha256()


def test_execution_record_round_trip() -> None:
    """Test that ExecutionRecord round trips through canonical JSON and dictionary."""
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
    record = ExecutionRecord(
        execution_id="exec-123",
        configuration_hash="a" * 64,
        benchmark="gsm8k",
        agent="agent-bob",
        task_id="task-456",
        status="SUCCESS",
        start_time=start,
        end_time=end,
        runtime_seconds=10.0,
        stdout="stdout",
        stderr="stderr",
        error_message="some error info",
        raw_output={"result": "value"},
        metadata={"nested": {"data": 123}}
    )
    
    # Dict serialization
    dumped_dict = record.canonical_dict()
    assert isinstance(dumped_dict, dict)
    assert isinstance(dumped_dict["start_time"], str)
    
    # Deserialization from dict
    restored_from_dict = ExecutionRecord.model_validate(dumped_dict)
    assert record == restored_from_dict

    # JSON serialization
    json_str = record.canonical_json()
    assert isinstance(json_str, str)
    
    # Deserialization from JSON
    restored_from_json = ExecutionRecord.from_canonical_json(json_str)
    assert record == restored_from_json
    assert record.sha256() == restored_from_json.sha256()


def test_execution_record_canonical_json() -> None:
    """Test that canonical JSON has sorted keys and contains compact separators."""
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
    record = ExecutionRecord(
        execution_id="exec-123",
        configuration_hash="a" * 64,
        benchmark="gsm8k",
        agent="agent-bob",
        task_id="task-456",
        status="SUCCESS",
        start_time=start,
        end_time=end,
        runtime_seconds=10.0,
        stdout="",
        stderr="",
        error_message=None,
        raw_output={},
        metadata={"b": 2, "a": 1}
    )
    json_str = record.canonical_json()
    
    # Compact format: no whitespace outside of values/keys
    assert " " not in json_str

    parsed = json.loads(json_str)
    sorted_keys = sorted(list(parsed.keys()))
    key_positions = [json_str.find(f'"{k}"') for k in sorted_keys]
    assert key_positions == sorted(key_positions)
