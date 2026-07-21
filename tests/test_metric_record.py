"""Tests for the new MetricRecord model (Artifact 6)."""

import json
import pytest
from pydantic import ValidationError
from records.metric_record import MetricRecord


def test_metric_record_instantiation() -> None:
    """Test that MetricRecord can be successfully instantiated with valid fields."""
    record = MetricRecord(
        metric_id="metric-123",
        evaluation_ids=["eval-1", "eval-2"],
        agent="agent-bob",
        benchmark="gsm8k",
        task_id="task-456",
        success_rate=0.8,
        repeated_run_consistency=0.9,
        perturbation_robustness=0.7,
        fault_tolerance=0.6,
        composite_reliability=0.75,
        metadata={"notes": "good performance"}
    )
    assert record.metric_id == "metric-123"
    assert record.evaluation_ids == ["eval-1", "eval-2"]
    assert record.agent == "agent-bob"
    assert record.benchmark == "gsm8k"
    assert record.task_id == "task-456"
    assert record.success_rate == 0.8
    assert record.repeated_run_consistency == 0.9
    assert record.perturbation_robustness == 0.7
    assert record.fault_tolerance == 0.6
    assert record.composite_reliability == 0.75
    assert record.metadata == {"notes": "good performance"}


def test_metric_record_immutability() -> None:
    """Test that MetricRecord is immutable and fields cannot be modified after creation."""
    record = MetricRecord(
        metric_id="metric-123",
        evaluation_ids=["eval-1"],
        agent="agent-bob",
        benchmark="gsm8k",
        task_id="task-456",
        success_rate=0.8,
        repeated_run_consistency=0.9,
        perturbation_robustness=0.7,
        fault_tolerance=0.6,
        composite_reliability=0.75,
        metadata={}
    )
    with pytest.raises(ValidationError):
        record.success_rate = 0.5  # type: ignore[misc]

    with pytest.raises(ValidationError):
        record.evaluation_ids = ["other"]  # type: ignore[misc]


def test_metric_record_rejects_unknown_fields() -> None:
    """Test that MetricRecord rejects unknown fields at instantiation."""
    with pytest.raises(ValidationError):
        MetricRecord(
            metric_id="metric-123",
            evaluation_ids=["eval-1"],
            agent="agent-bob",
            benchmark="gsm8k",
            task_id="task-456",
            success_rate=0.8,
            repeated_run_consistency=0.9,
            perturbation_robustness=0.7,
            fault_tolerance=0.6,
            composite_reliability=0.75,
            metadata={},
            extra_field="invalid"  # type: ignore[call-arg]
        )


def test_metric_record_empty_evaluation_ids() -> None:
    """Test that evaluation_ids cannot be empty."""
    with pytest.raises(ValidationError):
        MetricRecord(
            metric_id="metric-123",
            evaluation_ids=[],  # Invalid: empty list
            agent="agent-bob",
            benchmark="gsm8k",
            task_id="task-456",
            success_rate=0.8,
            repeated_run_consistency=0.9,
            perturbation_robustness=0.7,
            fault_tolerance=0.6,
            composite_reliability=0.75,
            metadata={}
        )


def test_metric_record_invalid_metric_values() -> None:
    """Test that metric values outside [0.0, 1.0] range raise ValidationError."""
    metrics_to_test = [
        "success_rate",
        "repeated_run_consistency",
        "perturbation_robustness",
        "fault_tolerance",
        "composite_reliability",
    ]
    base_args = {
        "metric_id": "metric-123",
        "evaluation_ids": ["eval-1"],
        "agent": "agent-bob",
        "benchmark": "gsm8k",
        "task_id": "task-456",
        "success_rate": 0.8,
        "repeated_run_consistency": 0.9,
        "perturbation_robustness": 0.7,
        "fault_tolerance": 0.6,
        "composite_reliability": 0.75,
    }
    
    for metric in metrics_to_test:
        # Test negative value
        invalid_args = base_args.copy()
        invalid_args[metric] = -0.1
        with pytest.raises(ValidationError):
            MetricRecord(**invalid_args)

        # Test value > 1.0
        invalid_args = base_args.copy()
        invalid_args[metric] = 1.0001
        with pytest.raises(ValidationError):
            MetricRecord(**invalid_args)


def test_metric_record_deterministic_hash() -> None:
    """Test that the SHA-256 hash is deterministic and identical for identical data."""
    record1 = MetricRecord(
        metric_id="metric-123",
        evaluation_ids=["eval-1", "eval-2"],
        agent="agent-bob",
        benchmark="gsm8k",
        task_id="task-456",
        success_rate=0.8,
        repeated_run_consistency=0.9,
        perturbation_robustness=0.7,
        fault_tolerance=0.6,
        composite_reliability=0.75,
        metadata={"a": 1, "b": 2}
    )
    record2 = MetricRecord(
        metric_id="metric-123",
        evaluation_ids=["eval-1", "eval-2"],
        agent="agent-bob",
        benchmark="gsm8k",
        task_id="task-456",
        success_rate=0.8,
        repeated_run_consistency=0.9,
        perturbation_robustness=0.7,
        fault_tolerance=0.6,
        composite_reliability=0.75,
        metadata={"b": 2, "a": 1}
    )
    assert record1.sha256() == record2.sha256()

    record3 = MetricRecord(
        metric_id="metric-123",
        evaluation_ids=["eval-1", "eval-2"],
        agent="agent-bob",
        benchmark="gsm8k",
        task_id="task-456",
        success_rate=0.8,
        repeated_run_consistency=0.9,
        perturbation_robustness=0.7,
        fault_tolerance=0.6,
        composite_reliability=0.749,  # slightly different
        metadata={"a": 1, "b": 2}
    )
    assert record1.sha256() != record3.sha256()


def test_metric_record_round_trip() -> None:
    """Test that MetricRecord round trips through canonical JSON and dictionary."""
    record = MetricRecord(
        metric_id="metric-123",
        evaluation_ids=["eval-1"],
        agent="agent-bob",
        benchmark="gsm8k",
        task_id="task-456",
        success_rate=0.8,
        repeated_run_consistency=0.9,
        perturbation_robustness=0.7,
        fault_tolerance=0.6,
        composite_reliability=0.75,
        metadata={"details": {"a": 100}}
    )
    
    # Test dictionary round trip
    dumped_dict = record.canonical_dict()
    assert isinstance(dumped_dict, dict)
    restored_from_dict = MetricRecord.model_validate(dumped_dict)
    assert record == restored_from_dict

    # Test JSON round trip
    json_str = record.canonical_json()
    assert isinstance(json_str, str)
    restored_from_json = MetricRecord.from_canonical_json(json_str)
    assert record == restored_from_json
    assert record.sha256() == restored_from_json.sha256()


def test_metric_record_canonical_json() -> None:
    """Test that canonical JSON has sorted keys and contains compact separators."""
    record = MetricRecord(
        metric_id="metric-123",
        evaluation_ids=["eval-1"],
        agent="agent-bob",
        benchmark="gsm8k",
        task_id="task-456",
        success_rate=0.8,
        repeated_run_consistency=0.9,
        perturbation_robustness=0.7,
        fault_tolerance=0.6,
        composite_reliability=0.75,
        metadata={"b": 2, "a": 1}
    )
    json_str = record.canonical_json()
    
    # Compact format: no whitespace outside values
    assert " " not in json_str

    parsed = json.loads(json_str)
    sorted_keys = sorted(list(parsed.keys()))
    key_positions = [json_str.find(f'"{k}"') for k in sorted_keys]
    assert key_positions == sorted(key_positions)
