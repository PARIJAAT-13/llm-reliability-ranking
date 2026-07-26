"""Tests for the Reliability Metrics Engine and individual metric calculators."""

from llm_reliability.agents.mock_agent import MockAgent
from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
from llm_reliability.configs.config import Configuration
from llm_reliability.ranking.reliability_ranker import ReliabilityRanker
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.reliability.faults.manager import FaultManager
from llm_reliability.reliability.metrics import (
    ConsistencyMetricResult, FaultToleranceMetric, FaultToleranceMetricResult,
    PromptPerturbationRobustnessMetric, ReliabilityMetricReport,
    ReliabilityMetricsEngine, ReliabilityReportGenerator,
    RepeatedRunConsistencyMetric, RobustnessMetricResult)
from llm_reliability.reliability.perturbation.manager import \
    PerturbationManager
from llm_reliability.reliability.repeated_runner import RepeatedRunner


def test_consistency_metric_calculation():
    metric = RepeatedRunConsistencyMetric()
    assert metric.name == "repeated_run_consistency"
    assert metric.dimension == "consistency"

    execs = [
        ExecutionRecord(
            configuration_hash="a" * 64,
            seed=42,
            benchmark="MockBenchmark",
            agent="MockAgent",
            task_id="t1",
            run_index=0,
            runtime_seconds=1.0,
            timestamp="2026-01-01T00:00:00+00:00",
            status="success",
            agent_output="Answer 1",
        ),
        ExecutionRecord(
            configuration_hash="a" * 64,
            seed=42,
            benchmark="MockBenchmark",
            agent="MockAgent",
            task_id="t1",
            run_index=1,
            runtime_seconds=1.2,
            timestamp="2026-01-01T00:00:01+00:00",
            status="success",
            agent_output="Answer 1",
        ),
    ]

    evals = [
        EvaluationRecord.from_execution(
            execs[0], success=True, score=1.0, evaluated_at="2026-01-01T00:00:02+00:00"
        ),
        EvaluationRecord.from_execution(
            execs[1], success=True, score=1.0, evaluated_at="2026-01-01T00:00:03+00:00"
        ),
    ]

    res = metric.compute(execs, evals)
    assert isinstance(res, ConsistencyMetricResult)
    assert res.success_rate == 1.0
    assert res.response_agreement_rate == 1.0
    assert res.execution_variance == 0.0
    assert res.deterministic_consistency_score == 1.0


def test_robustness_metric_calculation():
    metric = PromptPerturbationRobustnessMetric()
    assert metric.name == "prompt_perturbation_robustness"
    assert metric.dimension == "robustness"

    exec_base = ExecutionRecord(
        configuration_hash="a" * 64,
        seed=42,
        benchmark="MockBenchmark",
        agent="MockAgent",
        task_id="t1",
        run_index=0,
        perturbation=None,
        runtime_seconds=1.0,
        timestamp="2026-01-01T00:00:00+00:00",
        status="success",
        agent_output="Answer 1",
    )
    exec_pert = ExecutionRecord(
        configuration_hash="a" * 64,
        seed=42,
        benchmark="MockBenchmark",
        agent="MockAgent",
        task_id="t1",
        run_index=1,
        perturbation="whitespace",
        runtime_seconds=1.1,
        timestamp="2026-01-01T00:00:01+00:00",
        status="success",
        agent_output="Answer 1",
    )

    eval_base = EvaluationRecord.from_execution(
        exec_base, success=True, score=1.0, evaluated_at="2026-01-01T00:00:02+00:00"
    )
    eval_pert = EvaluationRecord.from_execution(
        exec_pert, success=True, score=1.0, evaluated_at="2026-01-01T00:00:03+00:00"
    )

    res = metric.compute([exec_base, exec_pert], [eval_base, eval_pert])
    assert isinstance(res, RobustnessMetricResult)
    assert res.success_retention_rate == 1.0
    assert res.response_stability == 1.0
    assert res.perturbation_sensitivity == 0.0
    assert res.robustness_score == 1.0


def test_fault_tolerance_metric_calculation():
    metric = FaultToleranceMetric()
    assert metric.name == "fault_tolerance"
    assert metric.dimension == "fault_tolerance"

    exec_base = ExecutionRecord(
        configuration_hash="a" * 64,
        seed=42,
        benchmark="MockBenchmark",
        agent="MockAgent",
        task_id="t1",
        run_index=0,
        fault_injected=False,
        runtime_seconds=1.0,
        timestamp="2026-01-01T00:00:00+00:00",
        status="success",
    )
    exec_fault = ExecutionRecord(
        configuration_hash="a" * 64,
        seed=42,
        benchmark="MockBenchmark",
        agent="MockAgent",
        task_id="t1",
        run_index=1,
        fault_injected=True,
        runtime_seconds=1.5,
        timestamp="2026-01-01T00:00:01+00:00",
        status="success",
        environment_metadata={
            "fault_injection": {
                "retry_count": 1,
                "recovery_status": "success",
                "latency_seconds": 1.5,
            }
        },
    )

    eval_base = EvaluationRecord.from_execution(
        exec_base, success=True, score=1.0, evaluated_at="2026-01-01T00:00:02+00:00"
    )
    eval_fault = EvaluationRecord.from_execution(
        exec_fault, success=True, score=1.0, evaluated_at="2026-01-01T00:00:03+00:00"
    )

    res = metric.compute([exec_base, exec_fault], [eval_base, eval_fault])
    assert isinstance(res, FaultToleranceMetricResult)
    assert res.recovery_rate == 1.0
    assert res.failure_resilience == 1.0
    assert res.fault_tolerance_score == 1.0


def test_engine_handles_missing_records_gracefully():
    engine = ReliabilityMetricsEngine()

    exec_base = ExecutionRecord(
        configuration_hash="a" * 64,
        seed=42,
        benchmark="MockBenchmark",
        agent="MockAgent",
        task_id="t1",
        run_index=0,
        runtime_seconds=1.0,
        timestamp="2026-01-01T00:00:00+00:00",
        status="success",
    )
    eval_base = EvaluationRecord.from_execution(
        exec_base, success=True, score=1.0, evaluated_at="2026-01-01T00:00:02+00:00"
    )

    # Only baseline records, no perturbation or fault records
    out = engine.compute_all([exec_base], [eval_base])
    assert "per_agent" in out
    assert "MockAgent" in out["per_agent"]
    agent_summary = out["per_agent"]["MockAgent"]

    assert agent_summary.consistency.success_rate == 1.0
    # Missing perturbation & fault tolerance records logged warnings and populated warning metadata
    assert "warning" in agent_summary.robustness.metadata
    assert "warning" in agent_summary.fault_tolerance.metadata
    assert agent_summary.composite_score > 0.0


def test_end_to_end_engine_to_ranking_pipeline():
    cfg = Configuration(
        experiment_name="engine_test",
        benchmark="MockBenchmark",
        agent="MockAgent",
        llm="mock",
        prompt_version="v1",
        dataset_version="v1",
        seed=42,
        repetitions=2,
    )
    benchmark = MockBenchmark(config=cfg)
    benchmark.load()
    agent = MockAgent(config=cfg)
    task = benchmark.get_task("mock-task-0")

    runner = RepeatedRunner(config=cfg, benchmark=benchmark, agent=agent)
    rep_res = runner.run_repeated_task(agent, benchmark, task, repetitions=2)

    pert_manager = PerturbationManager(config=cfg)
    pert_res = pert_manager.run_perturbed_task(agent, benchmark, task)

    fault_manager = FaultManager(config=cfg)
    fault_res = fault_manager.run_fault_injected_task(agent, benchmark, task)

    all_execs = rep_res.execution_records + pert_res.execution_records + fault_res.execution_records
    all_evals = (
        rep_res.evaluation_records + pert_res.evaluation_records + fault_res.evaluation_records
    )

    engine = ReliabilityMetricsEngine()
    out = engine.compute_all(all_execs, all_evals)

    report = ReliabilityReportGenerator.generate_report(out)
    assert isinstance(report, ReliabilityMetricReport)
    assert len(report.metric_records) > 0

    md = report.to_markdown()
    assert "# Quantitative LLM Reliability Evaluation Report" in md
    assert "MockAgent" in md

    # Verify produced MetricRecord instances are ingestible by ranking pipeline
    ranker = ReliabilityRanker()
    ranking = ranker.rank(report.metric_records, computed_at="2026-01-01T00:00:00+00:00")
    assert ranking.ranking_type == "reliability"
    assert len(ranking.rankings) > 0
