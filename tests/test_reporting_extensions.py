from __future__ import annotations

import csv
import json
import pathlib
import tempfile

import pytest

from llm_reliability.reporting.report_generator import ReportGenerator
from llm_reliability.reporting.summary import ExperimentSummary
from llm_reliability.statistics.result_models import (
    ConfidenceIntervalResult,
    CorrelationResult,
    EffectSizeResult,
    HypothesisTestResult,
    StatisticalReport,
    SummaryStatistics,
)
from tests.ranking_test_helpers import create_mock_metric
from tests.statistics_test_helpers import create_mock_ranking


@pytest.fixture
def sample_summary():
    metrics = [
        create_mock_metric("Agent A", success_rate=0.8, consistency=0.9, benchmark="mock-bench"),
        create_mock_metric("Agent B", success_rate=0.6, consistency=0.7, benchmark="mock-bench"),
    ]
    ranking_s = create_mock_ranking(
        {"Agent A": 0.8, "Agent B": 0.6}, ranking_type="success", benchmark="mock-bench"
    )
    ranking_r = create_mock_ranking(
        {"Agent A": 0.9, "Agent B": 0.7}, ranking_type="reliability", benchmark="mock-bench"
    )

    report = StatisticalReport(
        summary_statistics={
            "ranking1": SummaryStatistics(
                mean=0.7,
                median=0.7,
                variance=0.01,
                std_dev=0.1,
                min_val=0.6,
                max_val=0.8,
                q1=0.65,
                q3=0.75,
                count=2,
            ),
        },
        correlations={
            "spearman": CorrelationResult(coefficient=0.8, p_value=0.01, method="Spearman"),
        },
        hypothesis_tests=[
            HypothesisTestResult(
                statistic=2.0,
                p_value=0.05,
                method="t-test",
                alternative="two-sided",
                assumptions_met=True,
            ),
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


class TestCSVExport:
    def test_csv_export_format(self, sample_summary, tmp_path):
        gen = ReportGenerator()
        gen.generate(sample_summary, tmp_path, formats=["markdown"])
        md_path = tmp_path / "report.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "Test Experiment" in content

    def test_report_generator_accepts_csv_via_export(self, sample_summary, tmp_path):
        import pandas as pd

        rows = [m.canonical_dict() for m in sample_summary.metrics]
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "metrics.csv"
        df.to_csv(csv_path, index=False)
        assert csv_path.exists()
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows_read = list(reader)
        assert len(rows_read) == 2
        assert rows_read[0]["agent"] == "Agent A"
        assert rows_read[1]["agent"] == "Agent B"

    def test_csv_contains_expected_columns(self, sample_summary, tmp_path):
        import pandas as pd

        rows = [m.canonical_dict() for m in sample_summary.metrics]
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "columns.csv"
        df.to_csv(csv_path, index=False)
        with csv_path.open(encoding="utf-8") as f:
            header = f.readline().strip()
        assert "agent" in header
        assert "success_rate" in header
        assert "composite_reliability" in header


class TestJSONExport:
    def test_json_export_format(self, sample_summary, tmp_path):
        data = {
            "experiment_id": sample_summary.experiment_id,
            "experiment_name": sample_summary.experiment_name,
            "metrics": [m.canonical_dict() for m in sample_summary.metrics],
            "rankings": [r.canonical_dict() for r in sample_summary.rankings],
        }
        json_path = tmp_path / "report.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        assert json_path.exists()
        with json_path.open(encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["experiment_id"] == "test-exp-001"
        assert len(loaded["metrics"]) == 2

    def test_json_roundtrip_preserves_values(self, sample_summary, tmp_path):
        data = {
            "metrics": [
                {"agent": m.agent, "success_rate": m.success_rate} for m in sample_summary.metrics
            ],
        }
        json_path = tmp_path / "roundtrip.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        with json_path.open(encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["metrics"][0]["agent"] == "Agent A"
        assert loaded["metrics"][0]["success_rate"] == 0.8
        assert loaded["metrics"][1]["agent"] == "Agent B"
        assert loaded["metrics"][1]["success_rate"] == 0.6

    def test_json_with_statistical_report(self, sample_summary, tmp_path):
        report = sample_summary.statistical_report
        data = {
            "correlations": {
                k: {"coefficient": v.coefficient, "p_value": v.p_value}
                for k, v in report.correlations.items()
            },
            "effect_sizes": [
                {"value": e.value, "method": e.method, "interpretation": e.interpretation}
                for e in report.effect_sizes
            ],
        }
        json_path = tmp_path / "stats.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        with json_path.open(encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["correlations"]["spearman"]["coefficient"] == 0.8
        assert loaded["effect_sizes"][0]["value"] == 0.5

    def test_json_handles_empty_metrics(self, tmp_path):
        summary = ExperimentSummary(
            experiment_id="empty",
            experiment_name="Empty",
            metrics=[],
            rankings=[],
        )
        data = {"metrics": [m.canonical_dict() for m in summary.metrics]}
        json_str = json.dumps(data)
        loaded = json.loads(json_str)
        assert loaded["metrics"] == []


class TestMarkdownExport:
    def test_markdown_export_format(self, sample_summary, tmp_path):
        gen = ReportGenerator()
        paths = gen.generate(sample_summary, tmp_path, formats=["markdown"])
        md_path = paths["markdown"]
        content = md_path.read_text(encoding="utf-8")
        assert content.startswith("#") or "Test Experiment" in content

    def test_markdown_contains_agent_names(self, sample_summary, tmp_path):
        gen = ReportGenerator()
        paths = gen.generate(sample_summary, tmp_path, formats=["markdown"])
        content = paths["markdown"].read_text(encoding="utf-8")
        assert "Agent A" in content
        assert "Agent B" in content

    def test_markdown_contains_statistics_section(self, sample_summary, tmp_path):
        gen = ReportGenerator()
        paths = gen.generate(sample_summary, tmp_path, formats=["markdown"])
        content = paths["markdown"].read_text(encoding="utf-8")
        assert "Hypothesis" in content or "Effect Size" in content

    def test_markdown_generated_file_is_not_empty(self, sample_summary, tmp_path):
        gen = ReportGenerator()
        paths = gen.generate(sample_summary, tmp_path, formats=["markdown"])
        size = paths["markdown"].stat().st_size
        assert size > 0

    def test_markdown_has_consistent_line_breaks(self, sample_summary, tmp_path):
        gen = ReportGenerator()
        paths = gen.generate(sample_summary, tmp_path, formats=["markdown"])
        content = paths["markdown"].read_text(encoding="utf-8")
        lines = content.splitlines()
        assert len(lines) >= 10


class TestHTMLExport:
    def test_html_export_format(self, sample_summary, tmp_path):
        gen = ReportGenerator()
        paths = gen.generate(sample_summary, tmp_path, formats=["html"])
        html_path = paths["html"]
        content = html_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content or "<html" in content

    def test_html_contains_experiment_name(self, sample_summary, tmp_path):
        gen = ReportGenerator()
        paths = gen.generate(sample_summary, tmp_path, formats=["html"])
        content = paths["html"].read_text(encoding="utf-8")
        assert "Test Experiment" in content

    def test_html_contains_metric_tables(self, sample_summary, tmp_path):
        gen = ReportGenerator()
        paths = gen.generate(sample_summary, tmp_path, formats=["html"])
        content = paths["html"].read_text(encoding="utf-8")
        assert "Agent A" in content
        assert "Agent B" in content

    def test_html_generated_file_is_not_empty(self, sample_summary, tmp_path):
        gen = ReportGenerator()
        paths = gen.generate(sample_summary, tmp_path, formats=["html"])
        size = paths["html"].stat().st_size
        assert size > 0

    def test_html_closing_tags_present(self, sample_summary, tmp_path):
        gen = ReportGenerator()
        paths = gen.generate(sample_summary, tmp_path, formats=["html"])
        content = paths["html"].read_text(encoding="utf-8")
        assert "</html>" in content or "</body>" in content


class TestReportGeneratorExtensions:
    def test_generate_all_formats_default(self, sample_summary, tmp_path):
        gen = ReportGenerator()
        paths = gen.generate(sample_summary, tmp_path)
        assert "markdown" in paths
        assert "latex" in paths
        assert "html" in paths

    def test_generate_single_format(self, sample_summary, tmp_path):
        gen = ReportGenerator()
        paths = gen.generate(sample_summary, tmp_path, formats=["html"])
        assert "html" in paths
        assert "markdown" not in paths
        assert "latex" not in paths

    def test_output_dir_created_automatically(self, sample_summary, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        gen = ReportGenerator()
        paths = gen.generate(sample_summary, nested, formats=["markdown"])
        assert nested.exists()
        assert paths["markdown"].exists()

    def test_returns_paths_dict(self, sample_summary, tmp_path):
        gen = ReportGenerator()
        paths = gen.generate(sample_summary, tmp_path, formats=["markdown", "html"])
        assert isinstance(paths, dict)
        assert isinstance(paths["markdown"], pathlib.Path)
        assert isinstance(paths["html"], pathlib.Path)
