"""Tests for report generator and formats."""

import tempfile
import pathlib
import pytest

from llm_reliability.reporting.summary import ExperimentSummary
from llm_reliability.reporting.report_generator import ReportGenerator
from tests.ranking_test_helpers import create_mock_metric
from tests.statistics_test_helpers import create_mock_ranking
from llm_reliability.statistics.result_models import (
    CorrelationResult,
    HypothesisTestResult,
    EffectSizeResult,
    ConfidenceIntervalResult,
    StatisticalReport,
    SummaryStatistics,
)


@pytest.fixture
def sample_summary():
    metrics = [
        create_mock_metric("Agent A", success_rate=0.8, consistency=0.9, benchmark="mock-bench"),
        create_mock_metric("Agent B", success_rate=0.6, consistency=0.7, benchmark="mock-bench"),
    ]
    ranking_s = create_mock_ranking({"Agent A": 0.8, "Agent B": 0.6}, ranking_type="success", benchmark="mock-bench")
    ranking_r = create_mock_ranking({"Agent A": 0.9, "Agent B": 0.7}, ranking_type="reliability", benchmark="mock-bench")
    
    report = StatisticalReport(
        summary_statistics={
            "ranking1": SummaryStatistics(mean=0.7, median=0.7, variance=0.01, std_dev=0.1, min_val=0.6, max_val=0.8, q1=0.65, q3=0.75, count=2),
            "ranking2": SummaryStatistics(mean=0.8, median=0.8, variance=0.01, std_dev=0.1, min_val=0.7, max_val=0.9, q1=0.75, q3=0.85, count=2),
        },
        correlations={
            "spearman": CorrelationResult(coefficient=0.8, p_value=0.01, method="Spearman"),
        },
        hypothesis_tests=[
            HypothesisTestResult(statistic=2.0, p_value=0.05, method="t-test", alternative="two-sided", assumptions_met=True),
        ],
        effect_sizes=[
            EffectSizeResult(value=0.5, method="cohens_d", interpretation="medium"),
        ],
        confidence_intervals={
            "differences": ConfidenceIntervalResult(lower=0.02, upper=0.18, confidence_level=0.95),
        },
    )

    return ExperimentSummary(
        experiment_id="test-exp-001",
        experiment_name="Test Experiment",
        metrics=metrics,
        rankings=[ranking_s, ranking_r],
        statistical_report=report,
        config_snapshot={"param1": "value1"},
        metadata={"run_by": "test_suite"},
    )


def test_report_generator(sample_summary):
    gen = ReportGenerator()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = pathlib.Path(tmpdir)
        
        # Generate reports in all formats
        paths = gen.generate(
            summary=sample_summary,
            output_dir=output_path,
            formats=["markdown", "latex", "html"],
            figure_dir="figures",
        )
        
        assert "markdown" in paths
        assert "latex" in paths
        assert "html" in paths
        
        assert paths["markdown"].exists()
        assert paths["latex"].exists()
        assert paths["html"].exists()
        
        # Read files and verify key contents
        md_content = paths["markdown"].read_text(encoding="utf-8")
        assert "Test Experiment" in md_content
        assert "Agent A" in md_content
        assert "Hypothesis Tests" in md_content
        
        tex_content = paths["latex"].read_text(encoding="utf-8")
        assert "\\documentclass" in tex_content
        assert "Test Experiment" in tex_content
        
        html_content = paths["html"].read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html_content
        assert "Test Experiment" in html_content
