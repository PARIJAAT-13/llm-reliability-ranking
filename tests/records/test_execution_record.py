"""Tests for ExecutionRecord (Artifact 4)."""

import pytest
from pydantic import ValidationError

from llm_reliability.records import ExecutionRecord
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


def test_serialization_round_trip() -> None:
    record = _make_execution()
    restored = ExecutionRecord.from_canonical_json(record.canonical_json())
    assert record == restored


def test_hash_is_deterministic() -> None:
    assert _make_execution().sha256() == _make_execution().sha256()


def test_immutable() -> None:
    record = _make_execution()
    with pytest.raises(ValidationError):
        record.status = "failure"  # type: ignore[misc]


def test_rejects_invalid_configuration_hash_length() -> None:
    with pytest.raises(ValidationError):
        _make_execution(configuration_hash="short")


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ExecutionRecord(
            configuration_hash=CONFIG_HASH,
            seed=42,
            benchmark="mock",
            agent="mock_agent",
            task_id="task-1",
            run_index=0,
            runtime_seconds=1.0,
            timestamp=TIMESTAMP,
            status="success",
            extra=True,  # type: ignore[call-arg]
        )


def test_accepts_all_status_values() -> None:
    for status in ("success", "failure", "error", "timeout"):
        record = _make_execution(status=status)
        assert record.status == status
