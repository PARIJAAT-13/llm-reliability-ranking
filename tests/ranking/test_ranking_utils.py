"""Tests for ranking utility functions (validate_metrics, sort_and_rank)."""

import pytest

from llm_reliability.ranking.utils import sort_and_rank, validate_metrics
from tests.ranking_test_helpers import create_mock_metric


class TestValidateMetrics:
    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="Metrics list cannot be empty"):
            validate_metrics([])

    def test_benchmark_mismatch_raises(self) -> None:
        metrics = [
            create_mock_metric("A", benchmark="bench1"),
            create_mock_metric("B", benchmark="bench2"),
        ]
        with pytest.raises(ValueError, match="Benchmark mismatch"):
            validate_metrics(metrics)

    def test_task_level_raises(self) -> None:
        metrics = [
            create_mock_metric("A", benchmark="mock"),
        ]
        metrics[0] = metrics[0].model_copy(update={"task_id": "task-1"})
        with pytest.raises(ValueError, match="task_id must be None"):
            validate_metrics(metrics)

    def test_duplicate_agent_raises(self) -> None:
        metrics = [
            create_mock_metric("A", benchmark="mock"),
            create_mock_metric("A", benchmark="mock"),
        ]
        with pytest.raises(ValueError, match="Duplicate agent"):
            validate_metrics(metrics)

    def test_valid_list_passes(self) -> None:
        metrics = [
            create_mock_metric("A", benchmark="mock"),
            create_mock_metric("B", benchmark="mock"),
        ]
        validate_metrics(metrics)


class TestSortAndRank:
    def test_sorts_by_score_descending(self) -> None:
        metrics = [
            create_mock_metric("A", success_rate=0.5, benchmark="mock"),
            create_mock_metric("B", success_rate=0.9, benchmark="mock"),
            create_mock_metric("C", success_rate=0.7, benchmark="mock"),
        ]
        rankings = sort_and_rank(metrics, lambda m: m.success_rate)
        assert rankings == (("B", 0.9), ("C", 0.7), ("A", 0.5))

    def test_ties_broken_lexicographically(self) -> None:
        metrics = [
            create_mock_metric("B", success_rate=0.8, benchmark="mock"),
            create_mock_metric("A", success_rate=0.8, benchmark="mock"),
            create_mock_metric("C", success_rate=0.8, benchmark="mock"),
        ]
        rankings = sort_and_rank(metrics, lambda m: m.success_rate)
        assert rankings == (("A", 0.8), ("B", 0.8), ("C", 0.8))

    def test_deterministic_ordering(self) -> None:
        metrics = [
            create_mock_metric("X", success_rate=0.6, benchmark="mock"),
            create_mock_metric("Y", success_rate=0.9, benchmark="mock"),
        ]
        assert sort_and_rank(metrics, lambda m: m.success_rate) == (
            ("Y", 0.9),
            ("X", 0.6),
        )

    def test_returns_tuple_of_tuples(self) -> None:
        metrics = [create_mock_metric("A", success_rate=0.5, benchmark="mock")]
        result = sort_and_rank(metrics, lambda m: m.success_rate)
        assert isinstance(result, tuple)
        assert result[0] == ("A", 0.5)
