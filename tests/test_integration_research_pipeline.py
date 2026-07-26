"""
End-to-End Integration Test — Research Pipeline.

Tests the complete research pipeline from MockAgent + MockBenchmark through
to statistical comparison and report generation.  Each assertion validates
the contract between pipeline stages, making this test both a correctness
check and living documentation of the full data flow.

Pipeline under test
-------------------
MockAgent + MockBenchmark
  → EvaluationRecord
  → MetricRecord (MetricRecord.from_evaluations)
  → RankingEngine (success + reliability rankings)
  → StatisticalEngine.analyze (Spearman, Kendall, hypothesis tests, CIs)
  → analyze_ranking_divergence (overlap, divergence, displacement)
  → ReliabilityMetricsEngine.compute_all
  → ReliabilityScoreCalculator.compute
  → ReportGenerator (Markdown, LaTeX, HTML)
  → Configuration with extension fields (reliability_weights, visualization, statistical)
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone

import pytest

# ---------------------------------------------------------------------------
# Framework imports
# ---------------------------------------------------------------------------
from llm_reliability.configs.config import (Configuration,
                                            ReliabilityWeightsConfig,
                                            StatisticalOptions,
                                            VisualizationOptions)
from llm_reliability.ranking.ranking_engine import RankingEngine
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord
from llm_reliability.reliability.metrics.engine import ReliabilityMetricsEngine
from llm_reliability.reliability.score_calculator import (
    ReliabilityScore, ReliabilityScoreCalculator, ReliabilityScoreReport)
from llm_reliability.reporting.report_generator import ReportGenerator
from llm_reliability.reporting.summary import ExperimentSummary
from llm_reliability.statistics.ranking_divergence import (
    RankingDivergenceResult, analyze_ranking_divergence,
    compute_rank_displacement, compute_ranking_divergence,
    compute_ranking_overlap)
from llm_reliability.statistics.statistical_engine import StatisticalEngine

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TIMESTAMP = "2026-01-01T00:00:00+00:00"
CONFIG_HASH = "a" * 64
BENCHMARK_A = "benchmark_alpha"
BENCHMARK_B = "benchmark_beta"
AGENT_A = "agent_alpha"
AGENT_B = "agent_beta"


def _make_exec(
    benchmark: str,
    agent: str,
    task_id: str,
    run_index: int,
    status: str = "success",
    runtime: float = 1.0,
) -> ExecutionRecord:
    return ExecutionRecord(
        configuration_hash=CONFIG_HASH,
        seed=42,
        benchmark=benchmark,
        agent=agent,
        task_id=task_id,
        run_index=run_index,
        runtime_seconds=runtime,
        timestamp=TIMESTAMP,
        status=status,
        agent_output="result",
    )


def _make_eval(
    exec_rec: ExecutionRecord,
    success: bool,
    score: float = 1.0,
    run_index: int | None = None,
) -> EvaluationRecord:
    ev = EvaluationRecord.from_execution(
        exec_rec, success=success, score=score, evaluated_at=TIMESTAMP
    )
    if run_index is not None:
        ev = ev.model_copy(update={"run_index": run_index})
    return ev


@pytest.fixture
def two_agent_evaluations() -> dict[str, list]:
    """Create 4 evaluations per agent across two tasks for BENCHMARK_A."""
    execs = []
    evals = []

    # Agent A — high success rate (4/4)
    for run_idx in range(4):
        ex = _make_exec(BENCHMARK_A, AGENT_A, f"t{run_idx}", run_index=run_idx)
        ev = _make_eval(ex, success=True)
        execs.append(ex)
        evals.append(ev)

    # Agent B — lower success rate (2/4)
    for run_idx in range(4):
        ex = _make_exec(BENCHMARK_A, AGENT_B, f"t{run_idx}", run_index=run_idx)
        ev = _make_eval(ex, success=run_idx < 2)
        execs.append(ex)
        evals.append(ev)

    return {"executions": execs, "evaluations": evals}


@pytest.fixture
def benchmark_level_metric_records(
    two_agent_evaluations: dict,
) -> list[MetricRecord]:
    """Derive benchmark-level MetricRecords for each agent."""
    now = datetime.now(timezone.utc).isoformat()
    records = []
    for agent in (AGENT_A, AGENT_B):
        agent_evals = [ev for ev in two_agent_evaluations["evaluations"] if ev.agent == agent]
        records.append(MetricRecord.from_evaluations(agent_evals, computed_at=now))
    return records


# ---------------------------------------------------------------------------
# Part 9 — Configuration extension fields
# ---------------------------------------------------------------------------


class TestConfigurationExtension:
    def test_default_extension_fields(self):
        """Configuration instantiates with sane defaults for new fields."""
        cfg = Configuration(
            experiment_name="test",
            benchmark=BENCHMARK_A,
            agent=AGENT_A,
            llm="mock",
            prompt_version="v1",
            dataset_version="1.0",
            seed=0,
            repetitions=3,
        )
        assert isinstance(cfg.reliability_weights, ReliabilityWeightsConfig)
        assert isinstance(cfg.visualization, VisualizationOptions)
        assert isinstance(cfg.statistical, StatisticalOptions)

    def test_custom_reliability_weights(self):
        """Custom weights accepted and validated."""
        cfg = Configuration(
            experiment_name="test",
            benchmark=BENCHMARK_A,
            agent=AGENT_A,
            llm="mock",
            prompt_version="v1",
            dataset_version="1.0",
            seed=0,
            repetitions=3,
            reliability_weights=ReliabilityWeightsConfig(
                consistency=0.5, robustness=0.3, fault_tolerance=0.2
            ),
        )
        assert cfg.reliability_weights.consistency == pytest.approx(0.5)
        assert cfg.reliability_weights.robustness == pytest.approx(0.3)
        assert cfg.reliability_weights.fault_tolerance == pytest.approx(0.2)

    def test_invalid_reliability_weights_raise(self):
        """Weights that don't sum to 1.0 are rejected."""
        with pytest.raises(Exception):
            ReliabilityWeightsConfig(consistency=0.5, robustness=0.5, fault_tolerance=0.5)

    def test_visualization_options_defaults(self):
        """VisualizationOptions defaults are populated."""
        opts = VisualizationOptions()
        assert opts.dpi == 150
        assert "png" in opts.formats

    def test_statistical_options_defaults(self):
        """StatisticalOptions defaults are populated."""
        opts = StatisticalOptions()
        assert opts.confidence_level == pytest.approx(0.95)
        assert opts.bootstrap_iterations == 1000
        assert opts.compute_divergence is True

    def test_config_version_bumped(self):
        """CONFIG_VERSION is now 1.1.0 after the Part 9 extension."""
        from llm_reliability.configs.config import CONFIG_VERSION

        major, minor, patch = CONFIG_VERSION.split(".")
        assert int(major) >= 1
        assert int(minor) >= 1  # bumped from 1.0.0 → 1.1.0

    def test_config_serialization_roundtrip(self):
        """Extended Configuration serialises and deserialises without loss."""
        cfg = Configuration(
            experiment_name="roundtrip",
            benchmark=BENCHMARK_A,
            agent=AGENT_A,
            llm="mock",
            prompt_version="v1",
            dataset_version="1.0",
            seed=1,
            repetitions=2,
            reliability_weights=ReliabilityWeightsConfig(
                consistency=0.4, robustness=0.4, fault_tolerance=0.2
            ),
        )
        restored = Configuration.from_canonical_json(cfg.canonical_json())
        assert restored.reliability_weights.consistency == pytest.approx(0.4)
        assert restored.statistical.confidence_level == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Part 5 — ReliabilityScoreCalculator
# ---------------------------------------------------------------------------


class TestReliabilityScoreCalculator:
    def test_default_equal_weights(self):
        """Default weights are 1/3 each."""
        calc = ReliabilityScoreCalculator()
        assert calc._weights.consistency == pytest.approx(1 / 3, abs=1e-9)
        assert calc._weights.robustness == pytest.approx(1 / 3, abs=1e-9)
        assert calc._weights.fault_tolerance == pytest.approx(1 / 3, abs=1e-9)

    def test_custom_dict_weights(self):
        """Weights can be supplied as a plain dict."""
        calc = ReliabilityScoreCalculator(
            weights={"consistency": 0.6, "robustness": 0.3, "fault_tolerance": 0.1}
        )
        assert calc._weights.consistency == pytest.approx(0.6)

    def test_invalid_dict_weights_raise(self):
        """Dict weights that don't sum to 1.0 raise ValueError."""
        with pytest.raises(Exception):
            ReliabilityScoreCalculator(
                weights={
                    "consistency": 0.9,
                    "robustness": 0.9,
                    "fault_tolerance": 0.9,
                }
            )

    def test_compute_produces_report(self, two_agent_evaluations):
        """compute() returns a ReliabilityScoreReport with all scopes."""
        engine = ReliabilityMetricsEngine()
        engine_output = engine.compute_all(
            two_agent_evaluations["executions"],
            two_agent_evaluations["evaluations"],
        )

        calc = ReliabilityScoreCalculator()
        report = calc.compute(engine_output)

        assert isinstance(report, ReliabilityScoreReport)
        assert AGENT_A in report.per_agent
        assert AGENT_B in report.per_agent
        assert BENCHMARK_A in report.per_benchmark
        assert report.overall is not None

    def test_composite_in_unit_interval(self, two_agent_evaluations):
        """All composite scores must lie within [0, 1]."""
        engine = ReliabilityMetricsEngine()
        engine_output = engine.compute_all(
            two_agent_evaluations["executions"],
            two_agent_evaluations["evaluations"],
        )
        calc = ReliabilityScoreCalculator()
        report = calc.compute(engine_output)

        for score in report.per_agent.values():
            assert 0.0 <= score.composite_score <= 1.0

        if report.overall:
            assert 0.0 <= report.overall.composite_score <= 1.0

    def test_higher_success_yields_higher_consistency(self, two_agent_evaluations):
        """Agent A (100% success) should have a higher consistency score than Agent B (50%)."""
        engine = ReliabilityMetricsEngine()
        engine_output = engine.compute_all(
            two_agent_evaluations["executions"],
            two_agent_evaluations["evaluations"],
        )
        calc = ReliabilityScoreCalculator()
        report = calc.compute(engine_output)

        assert (
            report.per_agent[AGENT_A].consistency_score
            >= report.per_agent[AGENT_B].consistency_score
        )

    def test_to_markdown_not_empty(self, two_agent_evaluations):
        """to_markdown() returns a non-trivial string."""
        engine = ReliabilityMetricsEngine()
        engine_output = engine.compute_all(
            two_agent_evaluations["executions"],
            two_agent_evaluations["evaluations"],
        )
        calc = ReliabilityScoreCalculator()
        report = calc.compute(engine_output)
        md = report.to_markdown()
        assert "Reliability Score Report" in md
        assert AGENT_A in md

    def test_weight_redistribution_when_no_fault_data(self, two_agent_evaluations):
        """When fault_tolerance data is absent, its weight is redistributed."""
        engine = ReliabilityMetricsEngine()
        engine_output = engine.compute_all(
            two_agent_evaluations["executions"],
            two_agent_evaluations["evaluations"],
        )
        # Use custom weights emphasising fault_tolerance
        calc = ReliabilityScoreCalculator(
            weights={"consistency": 0.2, "robustness": 0.2, "fault_tolerance": 0.6}
        )
        report = calc.compute(engine_output)
        # All composites should still be in [0, 1] after redistribution
        for score in report.per_agent.values():
            assert 0.0 <= score.composite_score <= 1.0


# ---------------------------------------------------------------------------
# Part 6 extension — Ranking Divergence
# ---------------------------------------------------------------------------


class TestRankingDivergence:
    def _make_ranking(
        self,
        rankings: tuple[tuple[str, float], ...],
        ranking_type: str = "success",
        benchmark: str = BENCHMARK_A,
    ) -> RankingRecord:
        rank_map = {agent: i + 1 for i, (agent, _) in enumerate(rankings)}
        return RankingRecord(
            ranking_type=ranking_type,
            benchmark=benchmark,
            rankings=rankings,
            rank_map=rank_map,
            computed_at=TIMESTAMP,
        )

    def test_identical_rankings_have_full_overlap(self):
        rankings = ((AGENT_A, 0.9), (AGENT_B, 0.7))
        r1 = self._make_ranking(rankings, "success")
        r2 = self._make_ranking(rankings, "reliability")

        assert compute_ranking_overlap(r1, r2) == pytest.approx(1.0)

    def test_reversed_rankings_have_zero_overlap(self):
        r1 = self._make_ranking(((AGENT_A, 0.9), (AGENT_B, 0.7)), "success")
        r2 = self._make_ranking(((AGENT_B, 0.9), (AGENT_A, 0.7)), "reliability")

        overlap = compute_ranking_overlap(r1, r2)
        assert overlap == pytest.approx(0.0)

    def test_divergence_is_complement_of_overlap(self):
        r1 = self._make_ranking(((AGENT_A, 0.9), (AGENT_B, 0.7)), "success")
        r2 = self._make_ranking(((AGENT_B, 0.9), (AGENT_A, 0.7)), "reliability")

        overlap = compute_ranking_overlap(r1, r2)
        divergence = compute_ranking_divergence(r1, r2)
        assert overlap + divergence == pytest.approx(1.0)

    def test_zero_displacement_for_identical_rankings(self):
        rankings = ((AGENT_A, 0.9), (AGENT_B, 0.7))
        r1 = self._make_ranking(rankings, "success")
        r2 = self._make_ranking(rankings, "reliability")

        mean_disp, max_disp = compute_rank_displacement(r1, r2)
        assert mean_disp == pytest.approx(0.0)
        assert max_disp == 0

    def test_displacement_for_reversed_rankings(self):
        r1 = self._make_ranking(((AGENT_A, 0.9), (AGENT_B, 0.7)), "success")
        r2 = self._make_ranking(((AGENT_B, 0.9), (AGENT_A, 0.7)), "reliability")

        mean_disp, max_disp = compute_rank_displacement(r1, r2)
        # Both agents shift by 1 position
        assert mean_disp == pytest.approx(1.0)
        assert max_disp == 1

    def test_analyze_returns_result_model(self):
        r1 = self._make_ranking(((AGENT_A, 0.9), (AGENT_B, 0.7)), "success")
        r2 = self._make_ranking(((AGENT_A, 0.8), (AGENT_B, 0.6)), "reliability")

        result = analyze_ranking_divergence(r1, r2)
        assert isinstance(result, RankingDivergenceResult)
        assert result.n_agents == 2
        assert result.benchmark == BENCHMARK_A
        assert result.ranking1_type == "success"
        assert result.ranking2_type == "reliability"

    def test_analyze_overlap_divergence_sum_to_one(self):
        r1 = self._make_ranking(((AGENT_A, 0.9), (AGENT_B, 0.7)), "success")
        r2 = self._make_ranking(((AGENT_B, 0.9), (AGENT_A, 0.7)), "reliability")

        result = analyze_ranking_divergence(r1, r2)
        assert result.overlap + result.divergence == pytest.approx(1.0)

    def test_single_agent_overlap_is_one(self):
        r1 = self._make_ranking(((AGENT_A, 0.9),), "success")
        r2 = self._make_ranking(((AGENT_A, 0.5),), "reliability")

        result = analyze_ranking_divergence(r1, r2)
        assert result.overlap == pytest.approx(1.0)
        assert result.mean_displacement == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Part 10 — End-to-end pipeline integration
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Validates the complete research pipeline data flow."""

    def _build_metric_records(self) -> list[MetricRecord]:
        now = datetime.now(timezone.utc).isoformat()
        records = []
        for agent, success_count, total in [
            (AGENT_A, 4, 4),
            (AGENT_B, 2, 4),
        ]:
            evals = []
            for i in range(total):
                ex = _make_exec(BENCHMARK_A, agent, f"t{i}", run_index=i)
                ev = _make_eval(ex, success=i < success_count)
                evals.append(ev)
            records.append(MetricRecord.from_evaluations(evals, computed_at=now))
        return records

    def test_metric_record_derived_correctly(self):
        """MetricRecords encode correct success rates."""
        records = self._build_metric_records()
        agent_map = {r.agent: r for r in records}

        assert agent_map[AGENT_A].success_rate == pytest.approx(1.0)
        assert agent_map[AGENT_B].success_rate == pytest.approx(0.5)

    def test_ranking_engine_produces_both_ranking_types(self):
        """RankingEngine.rank_success and rank_reliability both succeed."""
        records = self._build_metric_records()
        now = datetime.now(timezone.utc).isoformat()
        engine = RankingEngine(metrics=records)

        success_ranking = engine.rank_success(computed_at=now)
        reliability_ranking = engine.rank_reliability(computed_at=now)

        assert isinstance(success_ranking, RankingRecord)
        assert isinstance(reliability_ranking, RankingRecord)
        assert len(success_ranking.rankings) == 2
        assert len(reliability_ranking.rankings) == 2

    def test_agent_a_ranks_first_in_success(self):
        """Agent A (100% success) should rank #1 in success ranking."""
        records = self._build_metric_records()
        now = datetime.now(timezone.utc).isoformat()
        engine = RankingEngine(metrics=records)
        ranking = engine.rank_success(computed_at=now)

        top_agent = ranking.rankings[0][0]
        assert top_agent == AGENT_A

    def test_statistical_engine_analyze(self):
        """StatisticalEngine.analyze runs full analysis on success vs reliability rankings."""
        records = self._build_metric_records()
        now = datetime.now(timezone.utc).isoformat()
        ranking_engine = RankingEngine(metrics=records)

        success_ranking = ranking_engine.rank_success(computed_at=now)
        reliability_ranking = ranking_engine.rank_reliability(computed_at=now)

        stat_report = StatisticalEngine.analyze(success_ranking, reliability_ranking)

        assert "spearman" in stat_report.correlations
        assert "kendall_tau" in stat_report.correlations
        assert -1.0 <= stat_report.correlations["spearman"].coefficient <= 1.0
        assert len(stat_report.hypothesis_tests) >= 2
        assert len(stat_report.effect_sizes) >= 2
        assert "differences" in stat_report.confidence_intervals

    def test_ranking_divergence_in_pipeline(self):
        """Ranking divergence analysis integrates cleanly with pipeline output."""
        records = self._build_metric_records()
        now = datetime.now(timezone.utc).isoformat()
        ranking_engine = RankingEngine(metrics=records)

        success_ranking = ranking_engine.rank_success(computed_at=now)
        reliability_ranking = ranking_engine.rank_reliability(computed_at=now)

        divergence = analyze_ranking_divergence(success_ranking, reliability_ranking)

        assert isinstance(divergence, RankingDivergenceResult)
        assert 0.0 <= divergence.overlap <= 1.0
        assert 0.0 <= divergence.divergence <= 1.0
        assert divergence.overlap + divergence.divergence == pytest.approx(1.0)

    def test_reliability_metrics_engine_full_compute(self):
        """ReliabilityMetricsEngine.compute_all processes all records and returns structured output."""
        execs = []
        evals = []
        for agent, success in [(AGENT_A, True), (AGENT_B, False)]:
            for run in range(3):
                ex = _make_exec(BENCHMARK_A, agent, f"task_{run}", run_index=run)
                ev = _make_eval(ex, success=success)
                execs.append(ex)
                evals.append(ev)

        engine = ReliabilityMetricsEngine()
        output = engine.compute_all(execs, evals)

        assert "per_agent" in output
        assert "per_benchmark" in output
        assert "overall" in output
        assert "metric_records" in output
        assert AGENT_A in output["per_agent"]
        assert AGENT_B in output["per_agent"]

    def test_score_calculator_integrated_in_pipeline(self):
        """ReliabilityScoreCalculator consumes engine output and produces valid scores."""
        execs = []
        evals = []
        for agent, success in [(AGENT_A, True), (AGENT_B, False)]:
            for run in range(3):
                ex = _make_exec(BENCHMARK_A, agent, f"task_{run}", run_index=run)
                ev = _make_eval(ex, success=success)
                execs.append(ex)
                evals.append(ev)

        engine = ReliabilityMetricsEngine()
        engine_output = engine.compute_all(execs, evals)

        calc = ReliabilityScoreCalculator(
            weights={"consistency": 0.5, "robustness": 0.3, "fault_tolerance": 0.2}
        )
        score_report = calc.compute(engine_output)

        assert isinstance(score_report, ReliabilityScoreReport)
        for name, score in score_report.per_agent.items():
            assert isinstance(score, ReliabilityScore)
            assert 0.0 <= score.composite_score <= 1.0
            assert len(score.available_dimensions) >= 1

    def test_report_generator_writes_all_formats(self):
        """ReportGenerator produces Markdown, LaTeX, and HTML report files."""
        records = self._build_metric_records()
        now = datetime.now(timezone.utc).isoformat()
        ranking_engine = RankingEngine(metrics=records)

        rankings = [
            ranking_engine.rank_success(computed_at=now),
            ranking_engine.rank_reliability(computed_at=now),
        ]

        stat_report = StatisticalEngine.analyze(rankings[0], rankings[1])

        summary = ExperimentSummary(
            experiment_id="integration-test-001",
            experiment_name="End-to-End Integration Test",
            metrics=records,
            rankings=rankings,
            statistical_report=stat_report,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator()
            output_paths = gen.generate(
                summary,
                output_dir=tmpdir,
                formats=["markdown", "latex", "html"],
            )

            assert "markdown" in output_paths
            assert "latex" in output_paths
            assert "html" in output_paths

            for fmt, path in output_paths.items():
                assert path.exists(), f"Report file missing for format: {fmt}"
                assert path.stat().st_size > 0, f"Report file empty for format: {fmt}"

    def test_full_pipeline_data_flow(self):
        """Smoke test that the complete pipeline produces no exceptions."""
        # Step 1 — Build records
        records = self._build_metric_records()
        now = datetime.now(timezone.utc).isoformat()

        # Step 2 — Rank
        ranking_engine = RankingEngine(metrics=records)
        success_ranking = ranking_engine.rank_success(computed_at=now)
        reliability_ranking = ranking_engine.rank_reliability(computed_at=now)

        # Step 3 — Statistical analysis
        stat_report = StatisticalEngine.analyze(success_ranking, reliability_ranking)

        # Step 4 — Ranking divergence
        divergence = analyze_ranking_divergence(success_ranking, reliability_ranking)

        # Step 5 — Reliability metrics + score
        execs, evals = [], []
        for r in records:
            for i in range(3):
                ex = _make_exec(r.benchmark, r.agent, f"t{i}", run_index=i)
                ev = _make_eval(ex, success=r.success_rate >= 0.5)
                execs.append(ex)
                evals.append(ev)

        engine = ReliabilityMetricsEngine()
        engine_output = engine.compute_all(execs, evals)
        calc = ReliabilityScoreCalculator()
        score_report = calc.compute(engine_output)

        # Assertions
        assert stat_report is not None
        assert divergence.overlap + divergence.divergence == pytest.approx(1.0)
        assert score_report.overall is not None
        assert 0.0 <= score_report.overall.composite_score <= 1.0
