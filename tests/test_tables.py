"""Tests for table generation and export."""

import tempfile
import pathlib
import pytest
import pandas as pd

from llm_reliability.visualization.tables import TableGenerator
from llm_reliability.visualization.export import TableExporter
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


def test_table_generator():
    tg = TableGenerator()
    
    metrics = [
        create_mock_metric("Agent A", success_rate=0.8, consistency=0.9, benchmark="mock-bench"),
        create_mock_metric("Agent B", success_rate=0.6, consistency=0.7, benchmark="mock-bench"),
    ]
    
    # Test summary statistics / reliability metrics table
    df_metrics = tg.reliability_metrics_table(metrics)
    assert isinstance(df_metrics, pd.DataFrame)
    assert len(df_metrics) == 2
    assert "Agent" in df_metrics.columns
    assert "Composite Reliability" in df_metrics.columns

    # Test ranking table
    ranking_s = create_mock_ranking({"Agent A": 0.8, "Agent B": 0.6}, ranking_type="success", benchmark="mock-bench")
    ranking_r = create_mock_ranking({"Agent A": 0.9, "Agent B": 0.7}, ranking_type="reliability", benchmark="mock-bench")
    df_ranking = tg.ranking_table(ranking_s, ranking_r)
    assert isinstance(df_ranking, pd.DataFrame)
    assert len(df_ranking) == 2
    assert "Agent" in df_ranking.columns
    assert "Success Rank" in df_ranking.columns
    assert "Reliability Rank" in df_ranking.columns

    # Test statistical report tables
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
    
    df_hyp = tg.hypothesis_test_table(report)
    assert isinstance(df_hyp, pd.DataFrame)
    assert len(df_hyp) == 1
    
    df_eff = tg.effect_size_table(report)
    assert isinstance(df_eff, pd.DataFrame)
    assert len(df_eff) == 1
    
    df_ci = tg.confidence_interval_table(report)
    assert isinstance(df_ci, pd.DataFrame)
    assert len(df_ci) == 1

    df_corr = tg.correlation_table(report)
    assert isinstance(df_corr, pd.DataFrame)
    assert len(df_corr) == 1

    # Test benchmark and agent summaries
    df_bench_sum = tg.benchmark_summary_table(metrics)
    assert len(df_bench_sum) == 1
    
    df_agent_sum = tg.agent_summary_table(metrics)
    assert len(df_agent_sum) == 2


def test_table_exporter():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "test_table"
        
        # Test save CSV, JSON, Markdown, LaTeX
        csv_path = TableExporter.save_csv(df, path)
        assert csv_path.exists()
        assert csv_path.suffix == ".csv"
        
        json_path = TableExporter.save_json(df, path)
        assert json_path.exists()
        assert json_path.suffix == ".json"
        
        md_path = TableExporter.save_markdown(df, path)
        assert md_path.exists()
        assert md_path.suffix == ".md"
        
        tex_path = TableExporter.save_latex(df, path)
        assert tex_path.exists()
        assert tex_path.suffix == ".tex"
        
        # Test save_all
        paths = TableExporter.save_all(df, path, skip_excel=True)
        assert "csv" in paths
        assert "json" in paths
        assert "markdown" in paths
        assert "latex" in paths
        for k, p in paths.items():
            assert p.exists()
