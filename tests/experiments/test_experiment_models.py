"""Tests for experiment models (BenchmarkSpec, AgentSpec, ExperimentSpec, ExperimentStatus)."""

import pytest
from pydantic import ValidationError

from llm_reliability.experiments.experiment_models import (
    AgentSpec,
    BenchmarkSpec,
    ExperimentSpec,
    ExperimentState,
    ExperimentStatus,
)


class TestBenchmarkSpec:
    def test_valid(self) -> None:
        spec = BenchmarkSpec(name="gaia", dataset_path="data/gaia.json")
        assert spec.name == "gaia"
        assert spec.dataset_path == "data/gaia.json"

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkSpec(name="", dataset_path="data/gaia.json")

    def test_rejects_empty_dataset_path(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkSpec(name="gaia", dataset_path="")

    def test_default_metadata(self) -> None:
        spec = BenchmarkSpec(name="gaia", dataset_path="data/gaia.json")
        assert spec.adapter_metadata == {}

    def test_immutable(self) -> None:
        spec = BenchmarkSpec(name="gaia", dataset_path="data/gaia.json")
        with pytest.raises(ValidationError):
            spec.name = "changed"


class TestAgentSpec:
    def test_valid(self) -> None:
        spec = AgentSpec(name="mock_agent")
        assert spec.name == "mock_agent"

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            AgentSpec(name="")

    def test_default_metadata(self) -> None:
        spec = AgentSpec(name="mock_agent")
        assert spec.metadata == {}
        assert spec.agent_metadata == {}

    def test_immutable(self) -> None:
        spec = AgentSpec(name="mock_agent")
        with pytest.raises(ValidationError):
            spec.name = "changed"


class TestExperimentSpec:
    def make_spec(self, **overrides: object) -> ExperimentSpec:
        defaults = {
            "experiment_name": "test-exp",
            "benchmarks": [BenchmarkSpec(name="gaia", dataset_path="data/gaia.json")],
            "agents": [AgentSpec(name="mock_agent")],
            "seeds": [42],
        }
        defaults.update(overrides)
        return ExperimentSpec(**defaults)

    def test_default_values(self) -> None:
        spec = self.make_spec()
        assert spec.experiment_id is not None
        assert spec.repetitions == 1
        assert spec.perturbations == []
        assert not spec.fault_injection
        assert not spec.parallel
        assert spec.max_workers == 4
        assert spec.output_dir == "results"
        assert spec.llm == "mock"
        assert spec.prompt_version == "1"
        assert spec.dataset_version == "1"

    def test_rejects_empty_benchmarks(self) -> None:
        with pytest.raises(ValidationError):
            self.make_spec(benchmarks=[])

    def test_rejects_empty_agents(self) -> None:
        with pytest.raises(ValidationError):
            self.make_spec(agents=[])

    def test_rejects_empty_seeds(self) -> None:
        with pytest.raises(ValidationError):
            self.make_spec(seeds=[])

    def test_rejects_duplicate_benchmark_names(self) -> None:
        bs = BenchmarkSpec(name="gaia", dataset_path="data/gaia.json")
        with pytest.raises(ValidationError, match="Duplicate benchmark"):
            self.make_spec(benchmarks=[bs, bs])

    def test_rejects_duplicate_agents(self) -> None:
        ag = AgentSpec(name="mock_agent")
        with pytest.raises(ValidationError, match="Duplicate agent"):
            self.make_spec(agents=[ag, ag])

    def test_rejects_negative_seeds(self) -> None:
        with pytest.raises(ValidationError, match="Seeds must be non-negative"):
            self.make_spec(seeds=[-1])

    def test_serialization_round_trip(self) -> None:
        spec = self.make_spec()
        json_str = spec.canonical_json()
        restored = ExperimentSpec.from_canonical_json(json_str)
        assert spec == restored
        assert spec.sha256() == restored.sha256()

    def test_immutable(self) -> None:
        spec = self.make_spec()
        with pytest.raises(ValidationError):
            spec.experiment_name = "changed"

    def test_accepts_multiple_seeds(self) -> None:
        spec = self.make_spec(seeds=[1, 2, 3])
        assert spec.seeds == [1, 2, 3]


class TestExperimentStatus:
    def test_default_pending(self) -> None:
        status = ExperimentStatus(experiment_id="exp-1")
        assert status.state == ExperimentState.PENDING
        assert status.total_runs == 0
        assert status.completed_runs == 0
        assert status.failed_runs == 0

    def test_progress_fraction_zero(self) -> None:
        status = ExperimentStatus(experiment_id="exp-1")
        assert status.progress_fraction() == 0.0

    def test_progress_fraction(self) -> None:
        status = ExperimentStatus(
            experiment_id="exp-1",
            total_runs=10,
            completed_runs=5,
        )
        assert status.progress_fraction() == 0.5

    def test_progress_fraction_all_done(self) -> None:
        status = ExperimentStatus(
            experiment_id="exp-1",
            total_runs=10,
            completed_runs=10,
        )
        assert status.progress_fraction() == 1.0

    def test_mutable(self) -> None:
        status = ExperimentStatus(experiment_id="exp-1")
        status.state = ExperimentState.RUNNING
        assert status.state == ExperimentState.RUNNING

    def test_serialization_round_trip(self) -> None:
        status = ExperimentStatus(
            experiment_id="exp-1",
            total_runs=10,
            completed_runs=5,
            state=ExperimentState.RUNNING,
        )
        json_str = status.canonical_json()
        restored = ExperimentStatus.from_canonical_json(json_str)
        assert status.experiment_id == restored.experiment_id
        assert status.completed_runs == restored.completed_runs

    def test_errors_list(self) -> None:
        status = ExperimentStatus(experiment_id="exp-1")
        assert status.errors == []
        status.errors.append({"run": 1, "error": "timeout"})
        assert len(status.errors) == 1

    def test_timestamps(self) -> None:
        status = ExperimentStatus(experiment_id="exp-1")
        assert status.started_at is None
        assert status.completed_at is None
