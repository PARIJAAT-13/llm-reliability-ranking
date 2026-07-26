"""Tests for SeedManager, Scheduler, ResultManager, and ExperimentManager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_reliability.experiments.experiment_manager import ExperimentManager
from llm_reliability.experiments.experiment_models import (AgentSpec,
                                                           BenchmarkSpec,
                                                           ExperimentSpec,
                                                           ExperimentState)
from llm_reliability.experiments.result_manager import ResultManager
from llm_reliability.experiments.scheduler import RunDescriptor, Scheduler
from llm_reliability.experiments.seed_manager import SeedManager
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord
from tests.conftest import CONFIG_HASH, TIMESTAMP


def _make_spec(**overrides) -> ExperimentSpec:
    defaults = dict(
        experiment_name="lifecycle_test",
        benchmarks=[BenchmarkSpec(name="mock_bench", dataset_path="/data/test.json")],
        agents=[AgentSpec(name="test_agent")],
        seeds=[42],
        repetitions=1,
    )
    defaults.update(overrides)
    return ExperimentSpec(**defaults)


# ======================================================================
# SeedManager
# ======================================================================


class TestSeedManager:
    def test_seed_manager_set_seed(self):
        sm = SeedManager([42])
        seed = sm.derive(42, "bench", "agent", 0)
        assert isinstance(seed, int)
        assert 0 <= seed <= 0xFFFFFFFF

    def test_seed_manager_set_seed_none(self):
        with pytest.raises(ValueError, match="at least one"):
            SeedManager([])

    def test_seed_manager_negative_seed_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            SeedManager([-1])

    def test_seed_manager_multiple_seeds(self):
        sm = SeedManager([42, 99])
        seeds = sm.all_seeds_for("bench", "agent", 2)
        assert len(seeds) == 4
        assert seeds[0] != seeds[2]

    def test_seed_manager_reproducibility(self):
        sm1 = SeedManager([42])
        sm2 = SeedManager([42])
        for bench in ("b1", "b2"):
            for agent in ("a1", "a2"):
                for run in range(3):
                    s1 = sm1.derive(42, bench, agent, run)
                    s2 = sm2.derive(42, bench, agent, run)
                    assert s1 == s2

    def test_seed_manager_all_seeds_for(self):
        sm = SeedManager([10, 20])
        seeds = sm.all_seeds_for("bench", "agent", 3)
        assert len(seeds) == 6
        assert len(set(seeds)) == 6

    def test_seed_manager_derive_unique(self):
        sm = SeedManager([42])
        s1 = sm.derive(42, "bench", "agent", 0)
        s2 = sm.derive(42, "bench", "agent", 1)
        s3 = sm.derive(42, "other", "agent", 0)
        assert len({s1, s2, s3}) == 3

    def test_seed_manager_base_seeds_property(self):
        sm = SeedManager([1, 2, 3])
        assert sm.base_seeds == [1, 2, 3]
        sm.base_seeds.append(4)
        assert sm.base_seeds == [1, 2, 3]

    def test_seed_manager_reproducibility_random(self):
        import random

        sm1 = SeedManager([42])
        sm2 = SeedManager([42])
        s1 = sm1.derive(42, "bench", "agent", 0)
        s2 = sm2.derive(42, "bench", "agent", 0)
        random.seed(s1)
        vals1 = [random.random() for _ in range(5)]
        random.seed(s2)
        vals2 = [random.random() for _ in range(5)]
        assert vals1 == vals2


# ======================================================================
# Scheduler
# ======================================================================


class TestScheduler:
    def test_scheduler_build_run_queue_basic(self):
        spec = _make_spec(
            benchmarks=[BenchmarkSpec(name="b1", dataset_path="/d1")],
            agents=[AgentSpec(name="a1")],
            seeds=[42],
            repetitions=3,
        )
        sched = Scheduler(spec)
        queue = sched.build_run_queue()
        assert len(queue) == 3
        for i, desc in enumerate(queue):
            assert desc.benchmark_name == "b1"
            assert desc.agent_name == "a1"
            assert desc.run_index == i
            assert desc.base_seed == 42
            assert desc.dataset_path == "/d1"

    def test_scheduler_build_run_queue_empty(self):
        with pytest.raises(Exception):
            _make_spec(repetitions=0)

    def test_scheduler_build_run_queue_single_combination(self):
        spec = _make_spec(
            benchmarks=[BenchmarkSpec(name="b1", dataset_path="/d1")],
            agents=[AgentSpec(name="a1")],
            seeds=[42],
            repetitions=1,
        )
        sched = Scheduler(spec)
        queue = sched.build_run_queue()
        assert len(queue) == 1
        desc = queue[0]
        assert desc.benchmark_name == "b1"
        assert desc.agent_name == "a1"
        assert desc.base_seed == 42
        assert desc.run_index == 0

    def test_scheduler_build_run_queue_all_combinations(self):
        spec = _make_spec(
            benchmarks=[
                BenchmarkSpec(name="b1", dataset_path="/d1"),
                BenchmarkSpec(name="b2", dataset_path="/d2"),
            ],
            agents=[
                AgentSpec(name="a1"),
                AgentSpec(name="a2"),
            ],
            seeds=[10, 20],
            repetitions=2,
        )
        sched = Scheduler(spec)
        queue = sched.build_run_queue()
        expected = 2 * 2 * 2 * 2
        assert len(queue) == expected

        bench_agents = {(d.benchmark_name, d.agent_name) for d in queue}
        assert bench_agents == {("b1", "a1"), ("b1", "a2"), ("b2", "a1"), ("b2", "a2")}

    def test_scheduler_run_descriptor_frozen(self):
        spec = _make_spec()
        sched = Scheduler(spec)
        queue = sched.build_run_queue()
        desc = queue[0]
        with pytest.raises(AttributeError):
            desc.benchmark_name = "changed"

    def test_scheduler_total_runs(self):
        spec = _make_spec(
            benchmarks=[BenchmarkSpec(name="b1", dataset_path="/d1")],
            agents=[AgentSpec(name="a1")],
            seeds=[42],
            repetitions=5,
        )
        sched = Scheduler(spec)
        assert sched.total_runs() == 5

    def test_scheduler_total_runs_multiple(self):
        spec = _make_spec(
            benchmarks=[
                BenchmarkSpec(name="b1", dataset_path="/d1"),
                BenchmarkSpec(name="b2", dataset_path="/d2"),
            ],
            agents=[AgentSpec(name="a1"), AgentSpec(name="a2")],
            seeds=[1, 2, 3],
            repetitions=4,
        )
        sched = Scheduler(spec)
        assert sched.total_runs() == 2 * 2 * 3 * 4

    def test_scheduler_agent_identifier(self):
        spec = _make_spec(
            agents=[AgentSpec(name="custom_agent", metadata={"model": "gpt-4"})],
            repetitions=1,
        )
        sched = Scheduler(spec)
        queue = sched.build_run_queue()
        assert queue[0].agent_name == "custom_agent:gpt-4"


# ======================================================================
# ResultManager
# ======================================================================


class TestResultManager:
    def test_result_manager_initialization(self, tmp_path: Path):
        spec = _make_spec(experiment_name="rm_test")
        rm = ResultManager(spec, output_dir=str(tmp_path / "results"))
        assert rm.experiment_dir.exists()
        assert (rm.experiment_dir / "logs").exists()

    def test_result_manager_save_checkpoint(self, tmp_path: Path):
        spec = _make_spec(experiment_name="rm_checkpoint")
        rm = ResultManager(spec, output_dir=str(tmp_path / "results"))
        rm.save_checkpoint([0, 1, 2, 5])
        cp_path = rm.experiment_dir / "checkpoint.json"
        assert cp_path.exists()
        data = json.loads(cp_path.read_text(encoding="utf-8"))
        assert data["completed"] == [0, 1, 2, 5]

    def test_result_manager_load_checkpoint(self, tmp_path: Path):
        spec = _make_spec(experiment_name="rm_load_cp")
        rm = ResultManager(spec, output_dir=str(tmp_path / "results"))
        rm.save_checkpoint([1, 3, 7])
        loaded = rm.load_checkpoint()
        assert loaded == [1, 3, 7]

    def test_result_manager_load_missing_file(self, tmp_path: Path):
        spec = _make_spec(experiment_name="rm_missing")
        rm = ResultManager(spec, output_dir=str(tmp_path / "results"))
        loaded = rm.load_checkpoint()
        assert loaded == []

    def test_result_manager_clear_checkpoint(self, tmp_path: Path):
        spec = _make_spec(experiment_name="rm_clear")
        rm = ResultManager(spec, output_dir=str(tmp_path / "results"))
        rm.save_checkpoint([0, 1])
        assert rm.load_checkpoint() == [0, 1]
        rm.save_checkpoint([])
        assert rm.load_checkpoint() == []

    def test_result_manager_save_configuration(self, tmp_path: Path):
        spec = _make_spec(experiment_name="rm_save_cfg")
        rm = ResultManager(spec, output_dir=str(tmp_path / "results"))
        rm.save_configuration()
        cfg_path = rm.experiment_dir / "configuration.json"
        assert cfg_path.exists()

    def test_result_manager_save_executions(self, tmp_path: Path):
        spec = _make_spec(experiment_name="rm_save_execs")
        rm = ResultManager(spec, output_dir=str(tmp_path / "results"))
        execs = [
            ExecutionRecord(
                configuration_hash=CONFIG_HASH,
                seed=42,
                benchmark="mock",
                agent="test_agent",
                task_id="t1",
                run_index=0,
                runtime_seconds=0.5,
                timestamp=TIMESTAMP,
                stdout="ok",
                stderr="",
                status="success",
            )
        ]
        rm.save_executions(execs)
        path = rm.experiment_dir / "executions.json"
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert len(loaded) == 1
        assert loaded[0]["task_id"] == "t1"

    def test_result_manager_save_evaluations(self, tmp_path: Path):
        spec = _make_spec(experiment_name="rm_save_evals")
        rm = ResultManager(spec, output_dir=str(tmp_path / "results"))
        evals = [
            EvaluationRecord(
                execution_hash=CONFIG_HASH,
                configuration_hash=CONFIG_HASH,
                seed=42,
                benchmark="mock",
                agent="test_agent",
                task_id="t1",
                run_index=0,
                success=True,
                score=1.0,
                evaluated_at=TIMESTAMP,
            )
        ]
        rm.save_evaluations(evals)
        assert (rm.experiment_dir / "evaluations.json").exists()

    def test_result_manager_save_metrics(self, tmp_path: Path):
        spec = _make_spec(experiment_name="rm_save_metrics")
        rm = ResultManager(spec, output_dir=str(tmp_path / "results"))
        metrics = [
            MetricRecord(
                benchmark="mock",
                agent="test_agent",
                evaluation_count=1,
                success_rate=1.0,
                repeated_run_consistency=1.0,
                composite_reliability=1.0,
                computed_at=TIMESTAMP,
            )
        ]
        rm.save_metrics(metrics)
        assert (rm.experiment_dir / "metrics.json").exists()

    def test_result_manager_save_rankings(self, tmp_path: Path):
        spec = _make_spec(experiment_name="rm_save_ranks")
        rm = ResultManager(spec, output_dir=str(tmp_path / "results"))
        rankings = [
            RankingRecord(
                benchmark="mock",
                ranking_type="success",
                rankings=(("agent-a", 0.9),),
                rank_map={"agent-a": 1},
                computed_at=TIMESTAMP,
            )
        ]
        rm.save_rankings(rankings)
        assert (rm.experiment_dir / "rankings.json").exists()

    def test_result_manager_save_metadata(self, tmp_path: Path):
        from llm_reliability.experiments.experiment_models import \
            ExperimentStatus

        spec = _make_spec(experiment_name="rm_save_meta")
        rm = ResultManager(spec, output_dir=str(tmp_path / "results"))
        status = ExperimentStatus(experiment_id=spec.experiment_id, state=ExperimentState.RUNNING)
        rm.save_metadata(status, extra={"note": "test run"})
        meta_path = rm.experiment_dir / "metadata.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["state"] == "running"
        assert data["note"] == "test run"

    def test_result_manager_save_and_load_executions(self, tmp_path: Path):
        spec = _make_spec(experiment_name="rm_saveload_exe")
        rm = ResultManager(spec, output_dir=str(tmp_path / "results"))
        execs = [
            ExecutionRecord(
                configuration_hash=CONFIG_HASH,
                seed=42,
                benchmark="mock",
                agent="test_agent",
                task_id="t1",
                run_index=0,
                runtime_seconds=0.5,
                timestamp=TIMESTAMP,
                stdout="ok",
                stderr="",
                status="success",
            )
        ]
        rm.save_executions(execs)
        loaded = rm.load_executions()
        assert len(loaded) == 1
        assert loaded[0]["task_id"] == "t1"

    def test_result_manager_load_missing_executions(self, tmp_path: Path):
        spec = _make_spec(experiment_name="rm_missing_exe")
        rm = ResultManager(spec, output_dir=str(tmp_path / "results"))
        assert rm.load_executions() == []

    def test_result_manager_save_evaluations_and_load(self, tmp_path: Path):
        spec = _make_spec(experiment_name="rm_evals_load")
        rm = ResultManager(spec, output_dir=str(tmp_path / "results"))
        evals = [
            EvaluationRecord(
                execution_hash=CONFIG_HASH,
                configuration_hash=CONFIG_HASH,
                seed=42,
                benchmark="mock",
                agent="test_agent",
                task_id="t1",
                run_index=0,
                success=True,
                score=1.0,
                evaluated_at=TIMESTAMP,
            )
        ]
        rm.save_evaluations(evals)
        loaded = rm.load_evaluations()
        assert len(loaded) == 1


# ======================================================================
# ExperimentManager
# ======================================================================


class TestExperimentManager:
    def test_experiment_manager_initialization(self, tmp_path: Path):
        workspace = tmp_path / "experiments"
        mgr = ExperimentManager(workspace)
        assert workspace.exists()
        assert mgr.list() == []

    def test_experiment_manager_create_experiment(self, tmp_path: Path):
        workspace = tmp_path / "experiments"
        mgr = ExperimentManager(workspace)
        spec = _make_spec(experiment_name="create_test")
        result = mgr.create(spec)
        assert result.experiment_id == spec.experiment_id
        file_path = workspace / f"{spec.experiment_id}.json"
        assert file_path.exists()

    def test_experiment_manager_create_duplicate(self, tmp_path: Path):
        workspace = tmp_path / "experiments"
        mgr = ExperimentManager(workspace)
        spec = _make_spec(experiment_name="dup_test")
        mgr.create(spec)
        with pytest.raises(FileExistsError):
            mgr.create(spec)

    def test_experiment_manager_load_experiment(self, tmp_path: Path):
        workspace = tmp_path / "experiments"
        mgr = ExperimentManager(workspace)
        spec = _make_spec(experiment_name="load_test")
        mgr.create(spec)
        loaded = mgr.load(spec.experiment_id)
        assert loaded.experiment_name == spec.experiment_name
        assert loaded.experiment_id == spec.experiment_id

    def test_experiment_manager_load_missing(self, tmp_path: Path):
        workspace = tmp_path / "experiments"
        mgr = ExperimentManager(workspace)
        with pytest.raises(FileNotFoundError):
            mgr.load("nonexistent_id")

    def test_experiment_manager_save_experiment(self, tmp_path: Path):
        workspace = tmp_path / "experiments"
        mgr = ExperimentManager(workspace)
        spec = _make_spec(experiment_name="save_test")
        mgr.create(spec)
        updated = spec.model_copy(update={"experiment_name": "renamed"})
        mgr.save(updated)
        loaded = mgr.load(spec.experiment_id)
        assert loaded.experiment_name == "renamed"

    def test_experiment_manager_list_experiments(self, tmp_path: Path):
        workspace = tmp_path / "experiments"
        mgr = ExperimentManager(workspace)
        s1 = _make_spec(experiment_name="list_test_a")
        s2 = _make_spec(experiment_name="list_test_b")
        mgr.create(s1)
        mgr.create(s2)
        ids = mgr.list()
        assert len(ids) == 2
        assert s1.experiment_id in ids
        assert s2.experiment_id in ids

    def test_experiment_manager_archive_experiment(self, tmp_path: Path):
        workspace = tmp_path / "experiments"
        archive_dir = tmp_path / "archive"
        mgr = ExperimentManager(workspace)
        spec = _make_spec(experiment_name="archive_test")
        mgr.create(spec)
        src_path = workspace / f"{spec.experiment_id}.json"
        assert src_path.exists()
        dst = mgr.archive(spec.experiment_id, archive_dir=str(archive_dir))
        assert not src_path.exists()
        assert dst.exists()
        assert spec.experiment_id not in mgr.list()

    def test_experiment_manager_archive_missing(self, tmp_path: Path):
        workspace = tmp_path / "experiments"
        mgr = ExperimentManager(workspace)
        with pytest.raises(FileNotFoundError):
            mgr.archive("nonexistent")

    def test_experiment_manager_empty_list(self, tmp_path: Path):
        workspace = tmp_path / "experiments"
        mgr = ExperimentManager(workspace)
        assert mgr.list() == []

    def test_experiment_manager_default_workspace(self):
        mgr = ExperimentManager()
        assert mgr._workspace.name == "experiments"
