"""Extended tests for experiment models — ExperimentSpec, RunDescriptor, seed derivation, and serialization."""

from __future__ import annotations

import json
import tempfile

import pytest

from llm_reliability.experiments.experiment_models import (
    AgentSpec,
    BenchmarkSpec,
    ExperimentSpec,
    ExperimentState,
    ExperimentStatus,
)
from llm_reliability.experiments.extended_models import (
    CheckpointState,
    ExperimentRunConfig,
    ModelGroup,
    ResourceLimits,
    SweepConfig,
    SweepMode,
    SweepParameter,
)
from llm_reliability.experiments.scheduler import RunDescriptor, Scheduler
from llm_reliability.experiments.seed_manager import SeedManager


class TestExperimentSpecCreation:
    def test_minimal_spec(self):
        spec = ExperimentSpec(
            experiment_name="minimal",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="data.json")],
            agents=[AgentSpec(name="mock")],
            seeds=[42],
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        assert spec.experiment_name == "minimal"
        assert len(spec.benchmarks) == 1
        assert len(spec.agents) == 1
        assert spec.seeds == [42]
        assert spec.repetitions == 1

    def test_spec_with_all_fields(self):
        spec = ExperimentSpec(
            experiment_name="full",
            benchmarks=[
                BenchmarkSpec(name="b1", dataset_path="d1.json", adapter_metadata={"type": "qa"})
            ],
            agents=[
                AgentSpec(name="a1", metadata={"model": "gpt-4"}, agent_metadata={"key": "val"})
            ],
            seeds=[42, 43, 44],
            repetitions=5,
            perturbations=["typo", "grammar"],
            fault_injection=True,
            parallel=True,
            max_workers=8,
            output_dir="custom_results",
            llm="gpt-4",
            prompt_version="v2",
            dataset_version="2.0",
            metadata={"note": "experiment note"},
        )
        assert spec.experiment_name == "full"
        assert spec.parallel is True
        assert spec.max_workers == 8
        assert spec.fault_injection is True
        assert len(spec.perturbations) == 2
        assert spec.output_dir == "custom_results"

    def test_spec_generates_experiment_id(self):
        spec1 = ExperimentSpec(
            experiment_name="id_test",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="data.json")],
            agents=[AgentSpec(name="mock")],
            seeds=[42],
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        spec2 = ExperimentSpec(
            experiment_name="id_test",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="data.json")],
            agents=[AgentSpec(name="mock")],
            seeds=[42],
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        assert spec1.experiment_id != spec2.experiment_id

    def test_spec_duplicate_benchmarks_raises(self):
        with pytest.raises(ValueError, match="Duplicate benchmark"):
            ExperimentSpec(
                experiment_name="dup_bench",
                benchmarks=[
                    BenchmarkSpec(name="mock", dataset_path="a.json"),
                    BenchmarkSpec(name="mock", dataset_path="b.json"),
                ],
                agents=[AgentSpec(name="mock")],
                seeds=[42],
                llm="mock",
                prompt_version="1",
                dataset_version="1",
            )

    def test_spec_duplicate_agents_raises(self):
        with pytest.raises(ValueError):
            ExperimentSpec(
                experiment_name="dup_agent",
                benchmarks=[BenchmarkSpec(name="mock", dataset_path="data.json")],
                agents=[AgentSpec(name="same_name"), AgentSpec(name="same_name")],
                seeds=[42],
                llm="mock",
                prompt_version="1",
                dataset_version="1",
            )

    def test_spec_negative_seeds_raises(self):
        with pytest.raises(ValueError, match="Seeds must be non-negative"):
            ExperimentSpec(
                experiment_name="neg_seed",
                benchmarks=[BenchmarkSpec(name="mock", dataset_path="data.json")],
                agents=[AgentSpec(name="mock")],
                seeds=[-5],
                llm="mock",
                prompt_version="1",
                dataset_version="1",
            )

    def test_spec_empty_seeds_raises(self):
        with pytest.raises(ValueError):
            ExperimentSpec(
                experiment_name="no_seeds",
                benchmarks=[BenchmarkSpec(name="mock", dataset_path="data.json")],
                agents=[AgentSpec(name="mock")],
                seeds=[],
                llm="mock",
                prompt_version="1",
                dataset_version="1",
            )

    def test_spec_empty_benchmarks_raises(self):
        with pytest.raises(ValueError):
            ExperimentSpec(
                experiment_name="no_bench",
                benchmarks=[],
                agents=[AgentSpec(name="mock")],
                seeds=[42],
                llm="mock",
                prompt_version="1",
                dataset_version="1",
            )

    def test_spec_empty_agents_raises(self):
        with pytest.raises(ValueError):
            ExperimentSpec(
                experiment_name="no_agent",
                benchmarks=[BenchmarkSpec(name="mock", dataset_path="data.json")],
                agents=[],
                seeds=[42],
                llm="mock",
                prompt_version="1",
                dataset_version="1",
            )


class TestExperimentSpecSerialization:
    def test_serialization_round_trip(self):
        spec = ExperimentSpec(
            experiment_name="roundtrip",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="data.json")],
            agents=[AgentSpec(name="mock")],
            seeds=[42],
            repetitions=3,
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        json_str = spec.canonical_json()
        restored = ExperimentSpec.from_canonical_json(json_str)
        assert restored.experiment_name == "roundtrip"
        assert restored.seeds == [42]
        assert restored.repetitions == 3
        assert len(restored.benchmarks) == 1
        assert len(restored.agents) == 1

    def test_serialization_with_metadata(self):
        spec = ExperimentSpec(
            experiment_name="meta_test",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="data.json")],
            agents=[AgentSpec(name="mock")],
            seeds=[42],
            llm="mock",
            prompt_version="1",
            dataset_version="1",
            metadata={"key": "value", "number": 42},
        )
        json_str = spec.canonical_json()
        restored = ExperimentSpec.from_canonical_json(json_str)
        assert restored.metadata["key"] == "value"
        assert restored.metadata["number"] == 42

    def test_serialization_with_perturbations(self):
        spec = ExperimentSpec(
            experiment_name="pert_test",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="data.json")],
            agents=[AgentSpec(name="mock")],
            seeds=[42],
            perturbations=["typo", "grammar", "spacing"],
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        json_str = spec.canonical_json()
        restored = ExperimentSpec.from_canonical_json(json_str)
        assert restored.perturbations == ["typo", "grammar", "spacing"]

    def test_serialization_preserves_output_dir(self):
        spec = ExperimentSpec(
            experiment_name="dir_test",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="data.json")],
            agents=[AgentSpec(name="mock")],
            seeds=[42],
            output_dir="/custom/path",
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        json_str = spec.canonical_json()
        restored = ExperimentSpec.from_canonical_json(json_str)
        assert restored.output_dir == "/custom/path"

    def test_serialization_from_file(self, tmp_path):
        spec = ExperimentSpec(
            experiment_name="file_test",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="data.json")],
            agents=[AgentSpec(name="mock")],
            seeds=[42],
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        path = tmp_path / "spec.json"
        path.write_text(spec.canonical_json(), encoding="utf-8")
        loaded = ExperimentSpec.from_canonical_json(path.read_text(encoding="utf-8"))
        assert loaded.experiment_name == "file_test"


class TestRunDescriptor:
    def test_run_descriptor_creation(self):
        desc = RunDescriptor(
            benchmark_name="bench_a",
            agent_name="agent_x",
            base_seed=42,
            run_index=0,
            derived_seed=12345,
            dataset_path="data.json",
        )
        assert desc.benchmark_name == "bench_a"
        assert desc.agent_name == "agent_x"
        assert desc.base_seed == 42
        assert desc.run_index == 0
        assert desc.derived_seed == 12345
        assert desc.dataset_path == "data.json"

    def test_run_descriptor_immutable(self):
        desc = RunDescriptor(
            benchmark_name="b",
            agent_name="a",
            base_seed=1,
            run_index=0,
            derived_seed=2,
            dataset_path="d.json",
        )
        with pytest.raises(AttributeError):
            desc.benchmark_name = "changed"

    def test_run_descriptor_in_frozen_dataclass(self):
        import dataclasses

        assert dataclasses.is_dataclass(RunDescriptor)


class TestSeedManager:
    def test_seed_derivation_consistency(self):
        manager = SeedManager([42])
        s1 = manager.derive(42, "bench_a", "agent_x", 0)
        s2 = manager.derive(42, "bench_a", "agent_x", 0)
        assert s1 == s2

    def test_seed_derivation_different_inputs(self):
        manager = SeedManager([42])
        s1 = manager.derive(42, "bench_a", "agent_x", 0)
        s2 = manager.derive(42, "bench_a", "agent_y", 0)
        assert s1 != s2

    def test_seed_derivation_different_benchmarks(self):
        manager = SeedManager([42])
        s1 = manager.derive(42, "bench_a", "agent_x", 0)
        s2 = manager.derive(42, "bench_b", "agent_x", 0)
        assert s1 != s2

    def test_seed_derivation_different_run_indices(self):
        manager = SeedManager([42])
        s1 = manager.derive(42, "bench_a", "agent_x", 0)
        s2 = manager.derive(42, "bench_a", "agent_x", 1)
        assert s1 != s2

    def test_seed_derivation_different_base_seeds(self):
        manager = SeedManager([42, 99])
        s1 = manager.derive(42, "bench_a", "agent_x", 0)
        s2 = manager.derive(99, "bench_a", "agent_x", 0)
        assert s1 != s2

    def test_seed_all_seeds_for(self):
        manager = SeedManager([42, 43])
        seeds = manager.all_seeds_for("bench_a", "agent_x", repetitions=3)
        assert len(seeds) == 2 * 3
        assert len(set(seeds)) == len(seeds)

    def test_seed_empty_raises(self):
        with pytest.raises(ValueError, match="requires at least one"):
            SeedManager([])

    def test_seed_negative_raises(self):
        with pytest.raises(ValueError, match="Seeds must be non-negative"):
            SeedManager([-1])

    def test_seed_base_seeds_property(self):
        manager = SeedManager([42, 99])
        assert manager.base_seeds == [42, 99]


class TestSchedulerRunQueue:
    def test_scheduler_single_run(self):
        spec = ExperimentSpec(
            experiment_name="single",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="d.json")],
            agents=[AgentSpec(name="test")],
            seeds=[42],
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        scheduler = Scheduler(spec)
        queue = scheduler.build_run_queue()
        assert len(queue) == 1
        assert queue[0].benchmark_name == "mock"
        assert queue[0].agent_name == "test"

    def test_scheduler_multiple_seeds(self):
        spec = ExperimentSpec(
            experiment_name="multi_seed",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="d.json")],
            agents=[AgentSpec(name="test")],
            seeds=[1, 2, 3],
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        scheduler = Scheduler(spec)
        queue = scheduler.build_run_queue()
        assert len(queue) == 3
        assert queue[0].base_seed == 1
        assert queue[1].base_seed == 2
        assert queue[2].base_seed == 3

    def test_scheduler_multiple_benchmarks(self):
        spec = ExperimentSpec(
            experiment_name="multi_bench",
            benchmarks=[
                BenchmarkSpec(name="b1", dataset_path="d1.json"),
                BenchmarkSpec(name="b2", dataset_path="d2.json"),
            ],
            agents=[AgentSpec(name="test")],
            seeds=[42],
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        scheduler = Scheduler(spec)
        queue = scheduler.build_run_queue()
        assert len(queue) == 2
        assert queue[0].benchmark_name == "b1"
        assert queue[1].benchmark_name == "b2"

    def test_scheduler_multiple_agents(self):
        spec = ExperimentSpec(
            experiment_name="multi_agent",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="d.json")],
            agents=[AgentSpec(name="a1"), AgentSpec(name="a2")],
            seeds=[42],
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        scheduler = Scheduler(spec)
        queue = scheduler.build_run_queue()
        assert len(queue) == 2
        assert queue[0].agent_name == "a1"
        assert queue[1].agent_name == "a2"

    def test_scheduler_repetitions(self):
        spec = ExperimentSpec(
            experiment_name="reps",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="d.json")],
            agents=[AgentSpec(name="test")],
            seeds=[42],
            repetitions=4,
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        scheduler = Scheduler(spec)
        queue = scheduler.build_run_queue()
        assert len(queue) == 4
        assert queue[0].run_index == 0
        assert queue[3].run_index == 3

    def test_scheduler_total_runs_calculation(self):
        spec = ExperimentSpec(
            experiment_name="calc",
            benchmarks=[
                BenchmarkSpec(name="b1", dataset_path="d1.json"),
                BenchmarkSpec(name="b2", dataset_path="d2.json"),
            ],
            agents=[AgentSpec(name="a1"), AgentSpec(name="a2"), AgentSpec(name="a3")],
            seeds=[1, 2],
            repetitions=3,
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        scheduler = Scheduler(spec)
        assert scheduler.total_runs() == 2 * 3 * 2 * 3


class TestExperimentStatus:
    def test_status_default_state(self):
        status = ExperimentStatus(experiment_id="test-123", total_runs=10)
        assert status.state == ExperimentState.PENDING
        assert status.completed_runs == 0
        assert status.failed_runs == 0

    def test_progress_fraction_zero(self):
        status = ExperimentStatus(experiment_id="test-123", total_runs=0)
        assert status.progress_fraction() == 0.0

    def test_progress_fraction_half(self):
        status = ExperimentStatus(experiment_id="test-123", total_runs=10, completed_runs=5)
        assert status.progress_fraction() == 0.5

    def test_progress_fraction_complete(self):
        status = ExperimentStatus(experiment_id="test-123", total_runs=10, completed_runs=10)
        assert status.progress_fraction() == 1.0

    def test_status_state_transitions(self):
        status = ExperimentStatus(experiment_id="test-123", total_runs=5)
        assert status.state == ExperimentState.PENDING
        status.state = ExperimentState.RUNNING
        assert status.state == ExperimentState.RUNNING
        status.state = ExperimentState.COMPLETED
        assert status.state == ExperimentState.COMPLETED
        status.state = ExperimentState.FAILED
        assert status.state == ExperimentState.FAILED

    def test_status_with_errors(self):
        status = ExperimentStatus(
            experiment_id="test-123",
            total_runs=5,
            failed_runs=2,
            errors=[{"run_index": 0, "error": "fail"}],
        )
        assert status.failed_runs == 2
        assert len(status.errors) == 1


class TestExtendedModels:
    def test_benchmark_spec_with_metadata(self):
        spec = BenchmarkSpec(
            name="arc", dataset_path="arc.json", adapter_metadata={"subset": "easy"}
        )
        assert spec.adapter_metadata["subset"] == "easy"

    def test_agent_spec_with_metadata(self):
        spec = AgentSpec(
            name="gpt", metadata={"model": "gpt-4"}, agent_metadata={"api_version": "2024"}
        )
        assert spec.metadata["model"] == "gpt-4"
        assert spec.agent_metadata["api_version"] == "2024"

    def test_sweep_parameter(self):
        param = SweepParameter(key="llm", values=["gpt-4", "gpt-4o"])
        assert param.key == "llm"
        assert param.values == ["gpt-4", "gpt-4o"]

    def test_sweep_config_product(self):
        config = SweepConfig(
            parameters=[SweepParameter(key="llm", values=["a", "b"])],
            mode=SweepMode.PRODUCT,
        )
        assert config.mode == SweepMode.PRODUCT

    def test_sweep_config_zip(self):
        config = SweepConfig(
            parameters=[SweepParameter(key="seed", values=[1, 2])],
            mode=SweepMode.ZIP,
        )
        assert config.mode == SweepMode.ZIP

    def test_model_group(self):
        group = ModelGroup(
            name="google_models",
            models=["gemini-2.0-flash", "gemini-2.5-pro"],
            runtime="gemini",
        )
        assert group.name == "google_models"
        assert len(group.models) == 2
        assert group.runtime == "gemini"

    def test_resource_limits_defaults(self):
        limits = ResourceLimits()
        assert limits.max_parallel_models == 1
        assert limits.runtime_timeout_seconds == 3600
        assert limits.log_rotation_max_mb == 100

    def test_checkpoint_state(self):
        state = CheckpointState(
            experiment_id="exp-1",
            total_runs=10,
            completed_indices=[0, 1, 2],
        )
        assert state.experiment_id == "exp-1"
        assert state.total_runs == 10
        assert len(state.completed_indices) == 3

    def test_checkpoint_state_serialization(self):
        state = CheckpointState(
            experiment_id="exp-1",
            total_runs=5,
            completed_indices=[0, 1],
            failed_indices=[2],
            started_at="2026-01-01T00:00:00",
        )
        json_str = state.canonical_json()
        restored = CheckpointState.from_canonical_json(json_str)
        assert restored.experiment_id == "exp-1"
        assert restored.completed_indices == [0, 1]
        assert restored.failed_indices == [2]

    def test_experiment_run_config(self):
        spec = ExperimentSpec(
            experiment_name="run_cfg",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="d.json")],
            agents=[AgentSpec(name="mock")],
            seeds=[42],
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        run_config = ExperimentRunConfig(
            experiment_spec=spec,
            runtime="mock",
            runtime_config={"temperature": 0.0},
        )
        assert run_config.runtime == "mock"
        assert run_config.runtime_config["temperature"] == 0.0
        assert run_config.repetition_count == 1
