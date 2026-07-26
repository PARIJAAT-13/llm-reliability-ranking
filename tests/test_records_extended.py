"""Extended tests for record models — 45+ tests."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from llm_reliability.records import (EvaluationRecord, ExecutionRecord,
                                     MetricRecord, RankingRecord)
from tests.conftest import CONFIG_HASH, TIMESTAMP


def _execution(**overrides: Any) -> ExecutionRecord:
    params: dict[str, Any] = dict(
        configuration_hash=CONFIG_HASH,
        seed=42,
        benchmark="mock",
        agent="test_agent",
        task_id="task-1",
        run_index=0,
        runtime_seconds=1.5,
        timestamp=TIMESTAMP,
        stdout="output",
        stderr="",
        status="success",
    )
    params.update(overrides)
    return ExecutionRecord(**params)


class TestExecutionRecord:
    def test_minimal_construction(self):
        rec = _execution()
        assert rec.configuration_hash == CONFIG_HASH
        assert rec.seed == 42
        assert rec.benchmark == "mock"
        assert rec.agent == "test_agent"
        assert rec.task_id == "task-1"
        assert rec.run_index == 0
        assert rec.status == "success"

    def test_default_fields(self):
        rec = _execution()
        assert rec.perturbation is None
        assert rec.fault_injected is False
        assert rec.error is None
        assert rec.agent_output is None
        assert rec.software_versions == {}
        assert rec.environment_metadata == {}

    def test_with_all_fields(self):
        rec = _execution(
            perturbation="typo",
            fault_injected=True,
            error="oom",
            agent_output={"result": "ok"},
            software_versions={"python": "3.11"},
            environment_metadata={"gpu": "A100"},
        )
        assert rec.perturbation == "typo"
        assert rec.fault_injected is True
        assert rec.error == "oom"
        assert rec.agent_output == {"result": "ok"}
        assert rec.software_versions == {"python": "3.11"}
        assert rec.environment_metadata == {"gpu": "A100"}

    def test_rejects_short_hash(self):
        with pytest.raises((TypeError, ValueError)):
            _execution(configuration_hash="tooshort")

    def test_rejects_empty_benchmark(self):
        with pytest.raises((TypeError, ValueError)):
            _execution(benchmark="")

    def test_rejects_negative_seed(self):
        with pytest.raises((TypeError, ValueError)):
            _execution(seed=-1)

    def test_rejects_negative_run_index(self):
        with pytest.raises((TypeError, ValueError)):
            _execution(run_index=-1)

    def test_rejects_negative_runtime(self):
        with pytest.raises((TypeError, ValueError)):
            _execution(runtime_seconds=-1.0)

    def test_rejects_invalid_status(self):
        with pytest.raises((TypeError, ValueError)):
            _execution(status="unknown_status")

    def test_accepts_all_valid_statuses(self):
        for status in ("success", "failure", "error", "timeout"):
            rec = _execution(status=status)
            assert rec.status == status

    def test_immutable(self):
        rec = _execution()
        with pytest.raises((TypeError, ValueError)):
            rec.status = "failure"

    def test_equality(self):
        a = _execution()
        b = _execution()
        assert a == b

    def test_inequality(self):
        a = _execution(task_id="task-1")
        b = _execution(task_id="task-2")
        assert a != b

    def test_sha256_consistent(self):
        rec = _execution()
        assert rec.sha256() == rec.sha256()

    def test_serialization_round_trip(self):
        rec = _execution()
        restored = ExecutionRecord.from_canonical_json(rec.canonical_json())
        assert rec == restored

    def test_hash_used_in_downstream(self):
        rec = _execution()
        assert len(rec.sha256()) == 64

    def test_stdout_stderr_empty(self):
        rec = _execution(stdout="", stderr="")
        assert rec.stdout == ""
        assert rec.stderr == ""

    def test_runtime_seconds_zero(self):
        rec = _execution(runtime_seconds=0.0)
        assert rec.runtime_seconds == 0.0


class TestEvaluationRecord:
    def test_from_execution(self):
        ex = _execution()
        ev = EvaluationRecord.from_execution(
            ex,
            success=True,
            score=1.0,
            evaluated_at=TIMESTAMP,
        )
        assert ev.execution_hash == ex.sha256()
        assert ev.configuration_hash == ex.configuration_hash
        assert ev.seed == ex.seed
        assert ev.benchmark == ex.benchmark
        assert ev.agent == ex.agent
        assert ev.task_id == ex.task_id
        assert ev.run_index == ex.run_index
        assert ev.perturbation == ex.perturbation
        assert ev.fault_injected == ex.fault_injected
        assert ev.success is True
        assert ev.score == 1.0
        assert ev.metrics == {}

    def test_from_execution_with_metrics(self):
        ex = _execution()
        ev = EvaluationRecord.from_execution(
            ex,
            success=True,
            score=0.95,
            metrics={"accuracy": 0.95},
            evaluated_at=TIMESTAMP,
        )
        assert ev.metrics == {"accuracy": 0.95}

    def test_from_execution_unsuccessful(self):
        ex = _execution()
        ev = EvaluationRecord.from_execution(
            ex,
            success=False,
            score=0.0,
            evaluated_at=TIMESTAMP,
        )
        assert ev.success is False
        assert ev.score == 0.0

    def test_direct_construction(self):
        ex = _execution()
        ev = EvaluationRecord(
            execution_hash=ex.sha256(),
            configuration_hash=CONFIG_HASH,
            seed=42,
            benchmark="mock",
            agent="test_agent",
            task_id="task-1",
            run_index=0,
            success=True,
            score=1.0,
            evaluated_at=TIMESTAMP,
        )
        assert ev.execution_hash == ex.sha256()

    def test_rejects_negative_score(self):
        ex = _execution()
        with pytest.raises((TypeError, ValueError)):
            EvaluationRecord.from_execution(
                ex,
                success=True,
                score=-0.1,
                evaluated_at=TIMESTAMP,
            )

    def test_rejects_score_above_one(self):
        ex = _execution()
        with pytest.raises((TypeError, ValueError)):
            EvaluationRecord.from_execution(
                ex,
                success=True,
                score=1.1,
                evaluated_at=TIMESTAMP,
            )

    def test_equality(self):
        ex = _execution()
        a = EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP)
        b = EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP)
        assert a == b

    def test_serialization_round_trip(self):
        ex = _execution()
        ev = EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP)
        restored = EvaluationRecord.from_canonical_json(ev.canonical_json())
        assert ev == restored

    def test_perturbation_propagated(self):
        ex = _execution(perturbation="typo")
        ev = EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP)
        assert ev.perturbation == "typo"

    def test_fault_injected_propagated(self):
        ex = _execution(fault_injected=True)
        ev = EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP)
        assert ev.fault_injected is True


class TestMetricRecord:
    def test_from_evaluations_basic(self):
        ex = _execution()
        evals = [
            EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP),
        ]
        m = MetricRecord.from_evaluations(evals, computed_at=TIMESTAMP)
        assert m.benchmark == "mock"
        assert m.agent == "test_agent"
        assert m.evaluation_count == 1
        assert m.success_rate == 1.0
        assert m.composite_reliability >= 0.0

    def test_from_evaluations_partial_success(self):
        ex = _execution()
        evals = [
            EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP),
            EvaluationRecord.from_execution(ex, success=False, score=0.0, evaluated_at=TIMESTAMP),
        ]
        m = MetricRecord.from_evaluations(evals, computed_at=TIMESTAMP)
        assert m.success_rate == 0.5
        assert m.evaluation_count == 2

    def test_from_evaluations_with_task_id(self):
        ex = _execution()
        evals = [
            EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP),
        ]
        m = MetricRecord.from_evaluations(evals, task_id="task-1", computed_at=TIMESTAMP)
        assert m.task_id == "task-1"

    def test_from_evaluations_raises_on_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            MetricRecord.from_evaluations([], computed_at=TIMESTAMP)

    def test_from_evaluations_raises_on_mismatch(self):
        ex1 = _execution(benchmark="bench-a")
        ex2 = _execution(benchmark="bench-b")
        evals = [
            EvaluationRecord.from_execution(ex1, success=True, score=1.0, evaluated_at=TIMESTAMP),
            EvaluationRecord.from_execution(ex2, success=True, score=1.0, evaluated_at=TIMESTAMP),
        ]
        with pytest.raises(ValueError, match="same benchmark"):
            MetricRecord.from_evaluations(evals, computed_at=TIMESTAMP)

    def test_repeated_run_consistency(self):
        ex = _execution()
        evals = [
            EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP),
            EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP),
            EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP),
        ]
        m = MetricRecord.from_evaluations(evals, computed_at=TIMESTAMP)
        assert m.repeated_run_consistency == 1.0

    def test_composite_within_bounds(self):
        ex = _execution()
        evals = [
            EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP),
            EvaluationRecord.from_execution(ex, success=False, score=0.0, evaluated_at=TIMESTAMP),
        ]
        m = MetricRecord.from_evaluations(evals, computed_at=TIMESTAMP)
        assert 0.0 <= m.composite_reliability <= 1.0

    def test_serialization_round_trip(self):
        ex = _execution()
        evals = [
            EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP),
        ]
        m = MetricRecord.from_evaluations(evals, computed_at=TIMESTAMP)
        restored = MetricRecord.from_canonical_json(m.canonical_json())
        assert m == restored

    def test_perturbation_robustness_none_without_perturbations(self):
        ex = _execution()
        evals = [
            EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP),
        ]
        m = MetricRecord.from_evaluations(evals, computed_at=TIMESTAMP)
        assert m.perturbation_robustness is None

    def test_fault_tolerance_none_without_faults(self):
        ex = _execution()
        evals = [
            EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP),
        ]
        m = MetricRecord.from_evaluations(evals, computed_at=TIMESTAMP)
        assert m.fault_tolerance is None

    def test_isr_fields_none_without_faults(self):
        ex = _execution()
        evals = [
            EvaluationRecord.from_execution(ex, success=True, score=1.0, evaluated_at=TIMESTAMP),
        ]
        m = MetricRecord.from_evaluations(evals, computed_at=TIMESTAMP)
        assert m.isr_output is None
        assert m.isr_behavior is None
        assert m.isr_composite_val is None


class TestRankingRecord:
    def test_from_metrics_success_ranking(self):
        m1 = _make_metric(agent="agent-a", success_rate=0.9, composite_reliability=0.8)
        m2 = _make_metric(agent="agent-b", success_rate=0.7, composite_reliability=0.6)
        r = RankingRecord.from_metrics([m1, m2], ranking_type="success", computed_at=TIMESTAMP)
        assert r.ranking_type == "success"
        assert r.benchmark == "mock"
        assert r.rankings[0][0] == "agent-a"
        assert r.rankings[1][0] == "agent-b"
        assert r.rank_map["agent-a"] == 1
        assert r.rank_map["agent-b"] == 2

    def test_from_metrics_reliability_ranking(self):
        m1 = _make_metric(agent="agent-a", success_rate=0.9, composite_reliability=0.95)
        m2 = _make_metric(agent="agent-b", success_rate=0.9, composite_reliability=0.85)
        r = RankingRecord.from_metrics([m1, m2], ranking_type="reliability", computed_at=TIMESTAMP)
        assert r.rankings[0][0] == "agent-a"
        assert r.rankings[1][0] == "agent-b"

    def test_raises_on_empty_metrics(self):
        with pytest.raises(ValueError, match="at least one"):
            RankingRecord.from_metrics([], ranking_type="success", computed_at=TIMESTAMP)

    def test_raises_on_mismatched_benchmarks(self):
        m1 = _make_metric(agent="a", benchmark="b1")
        m2 = _make_metric(agent="b", benchmark="b2")
        with pytest.raises(ValueError, match="same benchmark"):
            RankingRecord.from_metrics([m1, m2], ranking_type="success", computed_at=TIMESTAMP)

    def test_raises_on_task_level_metrics(self):
        m = _make_metric(agent="a", task_id="task-1")
        with pytest.raises(ValueError, match="task_id must be None"):
            RankingRecord.from_metrics([m], ranking_type="success", computed_at=TIMESTAMP)

    def test_tie_breaking_by_agent_name(self):
        m1 = _make_metric(agent="alpha", success_rate=0.9)
        m2 = _make_metric(agent="beta", success_rate=0.9)
        r = RankingRecord.from_metrics([m2, m1], ranking_type="success", computed_at=TIMESTAMP)
        assert r.rankings[0][0] == "alpha"
        assert r.rankings[1][0] == "beta"

    def test_serialization_round_trip(self):
        m1 = _make_metric(agent="a", success_rate=0.9)
        m2 = _make_metric(agent="b", success_rate=0.8)
        r = RankingRecord.from_metrics([m1, m2], ranking_type="success", computed_at=TIMESTAMP)
        restored = RankingRecord.from_canonical_json(r.canonical_json())
        assert r == restored

    def test_ranking_type_literal(self):
        m = _make_metric(agent="a")
        RankingRecord.from_metrics([m], ranking_type="success", computed_at=TIMESTAMP)
        RankingRecord.from_metrics([m], ranking_type="reliability", computed_at=TIMESTAMP)
        with pytest.raises((TypeError, ValueError)):
            RankingRecord.from_metrics([m], ranking_type="invalid", computed_at=TIMESTAMP)

    def test_rank_map_order_consistency(self):
        m1 = _make_metric(agent="z", success_rate=0.5)
        m2 = _make_metric(agent="a", success_rate=0.9)
        r = RankingRecord.from_metrics([m1, m2], ranking_type="success", computed_at=TIMESTAMP)
        assert r.rank_map["a"] == 1
        assert r.rank_map["z"] == 2

    def test_rank_map_counts_match(self):
        agents = [f"agent-{i}" for i in range(5)]
        metrics = [_make_metric(agent=a, success_rate=1.0 - i * 0.1) for i, a in enumerate(agents)]
        r = RankingRecord.from_metrics(metrics, ranking_type="success", computed_at=TIMESTAMP)
        assert len(r.rank_map) == 5
        assert len(r.rankings) == 5


def _make_metric(**kwargs):
    params = dict(
        benchmark="mock",
        agent="test_agent",
        evaluation_count=1,
        success_rate=1.0,
        repeated_run_consistency=1.0,
        composite_reliability=1.0,
        computed_at=TIMESTAMP,
    )
    params.update(kwargs)
    return MetricRecord(**params)
