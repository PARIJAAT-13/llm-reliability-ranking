"""Tests for new character-level perturbation strategies and auto-statistics wiring."""

import pytest

from llm_reliability.reliability.perturbation.strategies import (
    KeyboardNoiseStrategy, TypoPerturbationStrategy, UnicodeHomoglyphStrategy)
from llm_reliability.statistics.auto_selection import auto_select
from llm_reliability.statistics.report import \
    generate_reliability_statistical_report

# ===================================================================
# TypoPerturbationStrategy
# ===================================================================


def test_typo_strategy_name():
    s = TypoPerturbationStrategy()
    assert s.name == "typo"


def test_typo_alters_prompt():
    s = TypoPerturbationStrategy()
    task = {"prompt": "The quick brown fox jumps over the lazy dog"}
    result = s.apply(task, seed=42)
    assert result["prompt"] != task["prompt"]


def test_typo_preserves_structure():
    s = TypoPerturbationStrategy()
    task = {"prompt": "The quick brown fox jumps", "task_id": "t1", "ground_truth": "ans"}
    result = s.apply(task, seed=42)
    assert result["task_id"] == "t1"
    assert result["ground_truth"] == "ans"
    assert "metadata" in result
    assert "perturbation" in result["metadata"]


def test_typo_short_prompt_unchanged():
    s = TypoPerturbationStrategy()
    task = {"prompt": "Hi"}
    result = s.apply(task, seed=42)
    assert result["prompt"] == "Hi"


def test_typo_deterministic():
    s = TypoPerturbationStrategy()
    task = {"prompt": "This is a longer test prompt for deterministic checking"}
    r1 = s.apply(task, seed=123)
    r2 = s.apply(task, seed=123)
    assert r1["prompt"] == r2["prompt"]


def test_typo_different_seeds_differ():
    s = TypoPerturbationStrategy()
    task = {"prompt": "This is a longer test prompt for seed variation checking " * 3}
    r1 = s.apply(task, seed=1)
    r2 = s.apply(task, seed=2)
    assert r1["prompt"] != r2["prompt"]


# ===================================================================
# UnicodeHomoglyphStrategy
# ===================================================================


def test_homoglyph_strategy_name():
    s = UnicodeHomoglyphStrategy()
    assert s.name == "unicode_homoglyph"


def test_homoglyph_replaces_chars():
    s = UnicodeHomoglyphStrategy()
    task = {"prompt": "hello world"}
    result = s.apply(task, seed=42)
    assert not result.get("prompt", "").startswith("h") or result["prompt"] != task["prompt"]
    assert "metadata" in result


def test_homoglyph_alters_prompt():
    s = UnicodeHomoglyphStrategy()
    task = {"prompt": "acceptable"}
    result = s.apply(task, seed=42)
    assert result["prompt"] != task["prompt"]


def test_homoglyph_contains_unicode():
    s = UnicodeHomoglyphStrategy()
    task = {"prompt": "acceptable"}
    result = s.apply(task, seed=42)
    has_non_ascii = any(ord(c) > 127 for c in result["prompt"])
    assert has_non_ascii


def test_homoglyph_preserves_metadata():
    s = UnicodeHomoglyphStrategy()
    task = {"prompt": "Test case", "task_id": "t1", "ground_truth": "True"}
    result = s.apply(task, seed=42)
    assert result["task_id"] == "t1"
    assert result["ground_truth"] == "True"


def test_homoglyph_empty_prompt():
    s = UnicodeHomoglyphStrategy()
    task = {"prompt": ""}
    result = s.apply(task, seed=42)
    assert result["prompt"] == ""


def test_homoglyph_deterministic():
    s = UnicodeHomoglyphStrategy()
    task = {"prompt": "This is a test for deterministic behavior"}
    r1 = s.apply(task, seed=99)
    r2 = s.apply(task, seed=99)
    assert r1["prompt"] == r2["prompt"]


# ===================================================================
# KeyboardNoiseStrategy
# ===================================================================


def test_keyboard_noise_name():
    s = KeyboardNoiseStrategy()
    assert s.name == "keyboard_noise"


def test_keyboard_noise_alters_prompt():
    s = KeyboardNoiseStrategy()
    task = {"prompt": "This is a test prompt for keyboard noise."}
    result = s.apply(task, seed=42)
    assert result["prompt"] != task["prompt"]


def test_keyboard_noise_preserves_metadata():
    s = KeyboardNoiseStrategy()
    task = {"prompt": "Test", "task_id": "t1"}
    result = s.apply(task, seed=42)
    assert result["task_id"] == "t1"


def test_keyboard_noise_longer_or_equal():
    s = KeyboardNoiseStrategy()
    task = {"prompt": "This is a test prompt for keyboard noise checking."}
    result = s.apply(task, seed=42)
    assert len(result["prompt"]) >= len(task["prompt"])


def test_keyboard_noise_deterministic():
    s = KeyboardNoiseStrategy()
    task = {"prompt": "Test deterministic behavior checking for keyboard noise."}
    r1 = s.apply(task, seed=77)
    r2 = s.apply(task, seed=77)
    assert r1["prompt"] == r2["prompt"]


def test_keyboard_noise_empty_prompt():
    s = KeyboardNoiseStrategy()
    task = {"prompt": ""}
    result = s.apply(task, seed=42)
    assert result["prompt"] == ""


# ===================================================================
# Auto-statistics wiring
# ===================================================================


def test_generate_statistical_report_with_metrics():
    from llm_reliability.records.metric import MetricRecord

    metrics = [
        MetricRecord(
            benchmark="GAIA",
            agent="agent_a",
            task_id=None,
            evaluation_count=5,
            success_rate=0.8,
            repeated_run_consistency=0.7,
            perturbation_robustness=0.6,
            fault_tolerance=None,
            isr_output=None,
            isr_behavior=None,
            isr_composite_val=None,
            composite_reliability=0.75,
            computed_at="2026-01-01T00:00:00+00:00",
        ),
        MetricRecord(
            benchmark="GAIA",
            agent="agent_b",
            task_id=None,
            evaluation_count=5,
            success_rate=0.6,
            repeated_run_consistency=0.5,
            perturbation_robustness=0.4,
            fault_tolerance=None,
            isr_output=None,
            isr_behavior=None,
            isr_composite_val=None,
            composite_reliability=0.55,
            computed_at="2026-01-01T00:00:00+00:00",
        ),
    ]
    report = generate_reliability_statistical_report(metrics)
    assert report["n_metrics"] == 2
    assert report["n_groups"] == 2
    assert "recommended_test" in report
    assert "test_result" in report
    assert report["test_result"]["is_significant"] is not None


def test_generate_statistical_report_empty():
    report = generate_reliability_statistical_report([])
    assert report["n_metrics"] == 0
    assert "warning" in report


def test_generate_statistical_report_single_group():
    from llm_reliability.records.metric import MetricRecord

    metrics = [
        MetricRecord(
            benchmark="GAIA",
            agent="agent_a",
            task_id=None,
            evaluation_count=3,
            success_rate=0.9,
            repeated_run_consistency=0.8,
            perturbation_robustness=None,
            fault_tolerance=None,
            isr_output=None,
            isr_behavior=None,
            isr_composite_val=None,
            composite_reliability=0.85,
            computed_at="2026-01-01T00:00:00+00:00",
        ),
    ]
    report = generate_reliability_statistical_report(metrics)
    assert report["n_groups"] == 1
    assert "warning" in report


def test_auto_select_returns_recommendation():
    samples = [[1.0, 0.8, 0.9], [0.6, 0.5, 0.7]]
    result = auto_select(samples)
    assert "recommended_test" in result
    assert "n_groups" in result


def test_auto_select_three_groups():
    samples = [[1.0, 0.9], [0.7, 0.6], [0.5, 0.4]]
    result = auto_select(samples)
    assert result["recommended_test"] in ("kruskal_wallis", "anova_oneway")
