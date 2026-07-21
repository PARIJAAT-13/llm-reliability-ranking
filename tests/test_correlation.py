"""
Tests for correlation calculations.
"""

import pytest

from llm_reliability.statistics.correlation import (
    compute_kendall_tau,
    compute_spearman,
)
from tests.statistics_test_helpers import create_mock_ranking


def test_correlation_perfect_agreement():
    r1 = create_mock_ranking({"a": 1.0, "b": 2.0, "c": 3.0})
    r2 = create_mock_ranking({"a": 1.0, "b": 2.0, "c": 3.0})
    
    spearman = compute_spearman(r1, r2)
    kendall = compute_kendall_tau(r1, r2)

    assert spearman.coefficient == 1.0
    assert kendall.coefficient == 1.0
    assert spearman.p_value <= 1.0
    assert kendall.p_value <= 1.0


def test_correlation_perfect_disagreement():
    r1 = create_mock_ranking({"a": 1.0, "b": 2.0, "c": 3.0})
    r2 = create_mock_ranking({"a": 3.0, "b": 2.0, "c": 1.0})
    
    spearman = compute_spearman(r1, r2)
    kendall = compute_kendall_tau(r1, r2)

    assert spearman.coefficient == -1.0
    assert kendall.coefficient == -1.0


def test_correlation_mismatched_lengths():
    r1 = create_mock_ranking({"a": 1.0, "b": 2.0})
    r2 = create_mock_ranking({"a": 1.0, "b": 2.0, "c": 3.0})

    with pytest.raises(ValueError, match="Mismatched ranking lengths"):
        compute_spearman(r1, r2)


def test_correlation_mismatched_agents():
    r1 = create_mock_ranking({"a": 1.0, "b": 2.0})
    r2 = create_mock_ranking({"a": 1.0, "c": 2.0})

    with pytest.raises(ValueError, match="must contain the exact same set of agents"):
        compute_spearman(r1, r2)


def test_correlation_empty_rankings():
    r1 = create_mock_ranking({})
    r2 = create_mock_ranking({})

    with pytest.raises(ValueError, match="cannot be empty"):
        compute_spearman(r1, r2)
