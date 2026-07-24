"""Tests for the new RankingRecord model (Artifact 7)."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from records.ranking_record import RankingRecord


def test_ranking_record_instantiation() -> None:
    """Test that RankingRecord can be successfully instantiated with valid fields."""
    created = datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)
    record = RankingRecord(
        ranking_id="rank-123",
        benchmark="gsm8k",
        ranking_name="Reliability Ranking",
        metric_ids=["metric-1", "metric-2"],
        ranking_method="composite_reliability_desc",
        rankings={"agent-bob": 1, "agent-alice": 2},
        scores={"agent-bob": 0.92, "agent-alice": 0.85},
        created_at=created,
        metadata={"run": "pilot"},
    )
    assert record.ranking_id == "rank-123"
    assert record.benchmark == "gsm8k"
    assert record.ranking_name == "Reliability Ranking"
    assert record.metric_ids == ["metric-1", "metric-2"]
    assert record.ranking_method == "composite_reliability_desc"
    assert record.rankings == {"agent-bob": 1, "agent-alice": 2}
    assert record.scores == {"agent-bob": 0.92, "agent-alice": 0.85}
    assert record.created_at == created
    assert record.metadata == {"run": "pilot"}


def test_ranking_record_immutability() -> None:
    """Test that RankingRecord is immutable and fields cannot be modified after creation."""
    created = datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)
    record = RankingRecord(
        ranking_id="rank-123",
        benchmark="gsm8k",
        ranking_name="Reliability Ranking",
        metric_ids=["metric-1"],
        ranking_method="composite_reliability_desc",
        rankings={"agent-bob": 1},
        scores={"agent-bob": 0.92},
        created_at=created,
        metadata={},
    )
    with pytest.raises(ValidationError):
        record.ranking_name = "Success Ranking"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        record.rankings = {"agent-bob": 2}  # type: ignore[misc]


def test_ranking_record_rejects_unknown_fields() -> None:
    """Test that RankingRecord rejects unknown fields at instantiation."""
    created = datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(ValidationError):
        RankingRecord(
            ranking_id="rank-123",
            benchmark="gsm8k",
            ranking_name="Reliability Ranking",
            metric_ids=["metric-1"],
            ranking_method="composite_reliability_desc",
            rankings={"agent-bob": 1},
            scores={"agent-bob": 0.92},
            created_at=created,
            metadata={},
            extra_field="invalid",  # type: ignore[call-arg]
        )


def test_ranking_record_empty_fields_validation() -> None:
    """Test that empty values for name, method, metric_ids, rankings, and scores fail."""
    created = datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)

    # Empty ranking_name
    with pytest.raises(ValidationError):
        RankingRecord(
            ranking_id="rank-123",
            benchmark="gsm8k",
            ranking_name="",  # Invalid
            metric_ids=["metric-1"],
            ranking_method="composite_reliability_desc",
            rankings={"agent-bob": 1},
            scores={"agent-bob": 0.92},
            created_at=created,
        )

    # Empty ranking_method
    with pytest.raises(ValidationError):
        RankingRecord(
            ranking_id="rank-123",
            benchmark="gsm8k",
            ranking_name="Reliability Ranking",
            metric_ids=["metric-1"],
            ranking_method="",  # Invalid
            rankings={"agent-bob": 1},
            scores={"agent-bob": 0.92},
            created_at=created,
        )

    # Empty metric_ids
    with pytest.raises(ValidationError):
        RankingRecord(
            ranking_id="rank-123",
            benchmark="gsm8k",
            ranking_name="Reliability Ranking",
            metric_ids=[],  # Invalid
            ranking_method="composite_reliability_desc",
            rankings={"agent-bob": 1},
            scores={"agent-bob": 0.92},
            created_at=created,
        )

    # Empty rankings
    with pytest.raises(ValidationError):
        RankingRecord(
            ranking_id="rank-123",
            benchmark="gsm8k",
            ranking_name="Reliability Ranking",
            metric_ids=["metric-1"],
            ranking_method="composite_reliability_desc",
            rankings={},  # Invalid
            scores={"agent-bob": 0.92},
            created_at=created,
        )

    # Empty scores
    with pytest.raises(ValidationError):
        RankingRecord(
            ranking_id="rank-123",
            benchmark="gsm8k",
            ranking_name="Reliability Ranking",
            metric_ids=["metric-1"],
            ranking_method="composite_reliability_desc",
            rankings={"agent-bob": 1},
            scores={},  # Invalid
            created_at=created,
        )


def test_ranking_record_invalid_rank_values() -> None:
    """Test that non-positive rank integers raise ValidationError."""
    created = datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)
    # Zero rank
    with pytest.raises(ValidationError):
        RankingRecord(
            ranking_id="rank-123",
            benchmark="gsm8k",
            ranking_name="Reliability Ranking",
            metric_ids=["metric-1"],
            ranking_method="composite_reliability_desc",
            rankings={"agent-bob": 0},  # Invalid: not positive
            scores={"agent-bob": 0.92},
            created_at=created,
        )

    # Negative rank
    with pytest.raises(ValidationError):
        RankingRecord(
            ranking_id="rank-123",
            benchmark="gsm8k",
            ranking_name="Reliability Ranking",
            metric_ids=["metric-1"],
            ranking_method="composite_reliability_desc",
            rankings={"agent-bob": -1},  # Invalid: not positive
            scores={"agent-bob": 0.92},
            created_at=created,
        )


def test_ranking_record_deterministic_hash() -> None:
    """Test that the SHA-256 hash is deterministic and identical for identical data."""
    created = datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)
    record1 = RankingRecord(
        ranking_id="rank-123",
        benchmark="gsm8k",
        ranking_name="Reliability Ranking",
        metric_ids=["metric-1", "metric-2"],
        ranking_method="composite_reliability_desc",
        rankings={"agent-bob": 1, "agent-alice": 2},
        scores={"agent-bob": 0.92, "agent-alice": 0.85},
        created_at=created,
        metadata={"a": 1, "b": 2},
    )
    # Different order in metadata dict should not affect hash since canonical JSON sorts keys
    record2 = RankingRecord(
        ranking_id="rank-123",
        benchmark="gsm8k",
        ranking_name="Reliability Ranking",
        metric_ids=["metric-1", "metric-2"],
        ranking_method="composite_reliability_desc",
        rankings={"agent-alice": 2, "agent-bob": 1},  # reversed dict keys
        scores={"agent-alice": 0.85, "agent-bob": 0.92},
        created_at=created,
        metadata={"b": 2, "a": 1},
    )
    assert record1.sha256() == record2.sha256()

    # Slight modification results in different hash
    record3 = RankingRecord(
        ranking_id="rank-123",
        benchmark="gsm8k",
        ranking_name="Reliability Ranking",
        metric_ids=["metric-1", "metric-2"],
        ranking_method="composite_reliability_desc",
        rankings={"agent-bob": 1, "agent-alice": 2},
        scores={"agent-bob": 0.93, "agent-alice": 0.85},  # modified score
        created_at=created,
        metadata={"a": 1, "b": 2},
    )
    assert record1.sha256() != record3.sha256()


def test_ranking_record_round_trip() -> None:
    """Test that RankingRecord round trips through canonical JSON and dictionary."""
    created = datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)
    record = RankingRecord(
        ranking_id="rank-123",
        benchmark="gsm8k",
        ranking_name="Reliability Ranking",
        metric_ids=["metric-1"],
        ranking_method="composite_reliability_desc",
        rankings={"agent-bob": 1},
        scores={"agent-bob": 0.92},
        created_at=created,
        metadata={"info": {"nested": "value"}},
    )

    # Dict serialization
    dumped_dict = record.canonical_dict()
    assert isinstance(dumped_dict, dict)
    assert isinstance(dumped_dict["created_at"], str)

    # Deserialization from dict
    restored_from_dict = RankingRecord.model_validate(dumped_dict)
    assert record == restored_from_dict

    # JSON serialization
    json_str = record.canonical_json()
    assert isinstance(json_str, str)

    # Deserialization from JSON
    restored_from_json = RankingRecord.from_canonical_json(json_str)
    assert record == restored_from_json
    assert record.sha256() == restored_from_json.sha256()


def test_ranking_record_canonical_json() -> None:
    """Test that canonical JSON has sorted keys and contains compact separators."""
    created = datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)
    record = RankingRecord(
        ranking_id="rank-123",
        benchmark="gsm8k",
        ranking_name="Reliability Ranking",
        metric_ids=["metric-1"],
        ranking_method="composite_reliability_desc",
        rankings={"agent-bob": 1},
        scores={"agent-bob": 0.92},
        created_at=created,
        metadata={"b": 2, "a": 1},
    )
    json_str = record.canonical_json()

    # Compact format: no spaces in separators
    assert ", " not in json_str
    assert ": " not in json_str

    parsed = json.loads(json_str)
    sorted_keys = sorted(list(parsed.keys()))
    key_positions = [json_str.find(f'"{k}"') for k in sorted_keys]
    assert key_positions == sorted(key_positions)
