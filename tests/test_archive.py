"""Tests for archive generation and completeness checks."""

import tempfile
import pathlib

from llm_reliability.reproducibility.archive import ArchiveBuilder
from llm_reliability.reproducibility.checklist import ReproducibilityChecklist
from llm_reliability.reporting.summary import ExperimentSummary
from tests.ranking_test_helpers import create_mock_metric
from tests.statistics_test_helpers import create_mock_ranking


def test_archive_builder():
    metrics = [
        create_mock_metric("Agent A", success_rate=0.8, consistency=0.9, benchmark="mock-bench"),
        create_mock_metric("Agent B", success_rate=0.6, consistency=0.7, benchmark="mock-bench"),
    ]
    ranking_s = create_mock_ranking({"Agent A": 0.8, "Agent B": 0.6}, ranking_type="success", benchmark="mock-bench")
    ranking_r = create_mock_ranking({"Agent A": 0.9, "Agent B": 0.7}, ranking_type="reliability", benchmark="mock-bench")
    
    from llm_reliability.records.execution import ExecutionRecord
    from llm_reliability.records.evaluation import EvaluationRecord

    execs = [
        ExecutionRecord(
            configuration_hash="a" * 64,
            seed=42,
            benchmark="mock-bench",
            agent="Agent A",
            task_id="task-1",
            run_index=0,
            runtime_seconds=1.0,
            timestamp="2026-01-01T00:00:00+00:00",
            stdout="",
            stderr="",
            status="success",
            agent_output="ans",
            software_versions={},
            environment_metadata={},
        )
    ]
    evals = [
        EvaluationRecord.from_execution(
            execs[0],
            success=True,
            score=1.0,
            evaluated_at="2026-01-01T01:00:00+00:00",
        )
    ]

    summary = ExperimentSummary(
        experiment_id="test-archive-789",
        experiment_name="Archive Verification Test",
        metrics=metrics,
        rankings=[ranking_s, ranking_r],
        executions=execs,
        evaluations=evals,
        config_snapshot={"test": True},
    )
    
    builder = ArchiveBuilder(matplotlib_backend="Agg")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_root = pathlib.Path(tmpdir)
        archive_dir = builder.build(summary, root_dir=dest_root, skip_excel=True)
        
        # Verify directories exist
        assert archive_dir.exists()
        assert (archive_dir / "figures").is_dir()
        assert (archive_dir / "tables").is_dir()
        assert (archive_dir / "reports").is_dir()
        
        # Verify required files exist
        assert (archive_dir / "manifest.json").exists()
        assert (archive_dir / "environment.json").exists()
        assert (archive_dir / "CITATION.cff").exists()
        assert (archive_dir / "CHECKLIST.md").exists()
        assert (archive_dir / "README.md").exists()
        
        # Verify some figures and tables are written
        assert (archive_dir / "figures" / "ranking_comparison.png").exists()
        assert (archive_dir / "tables" / "reliability_metrics.csv").exists()
        assert (archive_dir / "reports" / "report.md").exists()
        assert (archive_dir / "reports" / "report.html").exists()

        # Run reproducibility checklist verify
        checklist = ReproducibilityChecklist()
        res = checklist.run(summary, archive_dir=archive_dir)
        assert res.all_passed
        assert res.critical_passed
        assert res.n_passed == res.n_total
