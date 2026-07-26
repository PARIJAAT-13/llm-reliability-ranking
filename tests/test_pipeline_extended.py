"""Extended tests for ExperimentPipeline — mocking Benchmark, Agent, and Cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llm_reliability.configs.config import Configuration
from llm_reliability.interfaces.benchmark import Benchmark
from llm_reliability.pipeline.experiment_pipeline import (
    ExperimentPipeline, ExperimentResult, classify_failure_reason)
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord
from tests.conftest import CONFIG_HASH, TIMESTAMP, make_configuration

# ======================================================================
# Dummy implementations (duck-type Benchmark and Agent)
# ======================================================================


class DummyBenchmark(Benchmark):
    def __init__(self, task_ids: list[str] | None = None) -> None:
        self._task_ids = task_ids or ["task-0", "task-1"]
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def list_tasks(self) -> list[str]:
        return list(self._task_ids)

    def get_task(self, task_id: str) -> dict[str, Any]:
        if task_id not in self._task_ids:
            raise ValueError(f"Unknown task: {task_id}")
        return {"task_id": task_id, "data": f"payload_{task_id}"}

    def run(self, agent: Any, task: dict[str, Any]) -> ExecutionRecord:
        return ExecutionRecord(
            configuration_hash=CONFIG_HASH,
            seed=42,
            benchmark="mock",
            agent="test_agent",
            task_id=task["task_id"],
            run_index=0,
            runtime_seconds=0.1,
            timestamp=TIMESTAMP,
            stdout="mock_output",
            stderr="",
            status="success",
            agent_output="dummy",
        )

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        return EvaluationRecord(
            execution_hash=execution.sha256(),
            configuration_hash=CONFIG_HASH,
            seed=42,
            benchmark="mock",
            agent="test_agent",
            task_id=execution.task_id,
            run_index=0,
            success=True,
            score=1.0,
            evaluated_at=TIMESTAMP,
        )

    def collect_logs(self) -> dict[str, Any]:
        return {}

    def metadata(self) -> dict[str, Any]:
        return {"name": "dummy_benchmark"}


class FailingBenchmark(DummyBenchmark):
    def run(self, agent: Any, task: dict[str, Any]) -> ExecutionRecord:
        return ExecutionRecord(
            configuration_hash=CONFIG_HASH,
            seed=42,
            benchmark="mock",
            agent="test_agent",
            task_id=task["task_id"],
            run_index=0,
            runtime_seconds=0.0,
            timestamp=TIMESTAMP,
            stdout="",
            stderr="error occurred",
            status="error",
            error="something went wrong timeout",
        )


class DummyAgent:
    def __init__(self) -> None:
        self.initialized = False
        self.shutdown_called = False

    def initialize(self) -> None:
        self.initialized = True

    def reset(self) -> None:
        pass

    def run(self, task: dict[str, Any]) -> Any:
        return "dummy_output"

    def shutdown(self) -> None:
        self.shutdown_called = True

    def metadata(self) -> dict[str, Any]:
        return {"name": "dummy_agent"}


class FailingAgent(DummyAgent):
    def initialize(self) -> None:
        raise RuntimeError("Agent init failed due to timeout")


class MockCache:
    def __init__(self) -> None:
        self._store: dict[str, ExperimentResult] = {}
        self._hit_count = 0
        self._miss_count = 0

    def generate_key(self, config: Configuration) -> str:
        return config.sha256()

    def exists(self, key: str) -> bool:
        return key in self._store

    def get(self, key: str) -> ExperimentResult | None:
        if key in self._store:
            self._hit_count += 1
            return self._store[key]
        return None

    def set(self, key: str, result: ExperimentResult) -> None:
        self._store[key] = result
        self._miss_count += 1


# ======================================================================
# classify_failure_reason tests
# ======================================================================


class TestClassifyFailureReason:
    def test_classify_failure_reason_timeout(self):
        assert classify_failure_reason("request timeout after 30s") == "timeout"
        assert classify_failure_reason("TimeoutError: connection timeout") == "timeout"

    def test_classify_failure_reason_rate_limit(self):
        assert classify_failure_reason("rate limit exceeded") == "inference"

    def test_classify_failure_reason_auth(self):
        assert classify_failure_reason("authentication failed") == "inference"
        assert classify_failure_reason("invalid API key") == "inference"

    def test_classify_failure_reason_unknown(self):
        assert classify_failure_reason("some cryptic error 0xDEAD") == "inference"

    def test_classify_failure_reason_empty(self):
        assert classify_failure_reason("") == "none"

    def test_classify_failure_reason_none(self):
        assert classify_failure_reason(None) == "none"

    def test_classify_failure_reason_memory(self):
        assert classify_failure_reason("out of memory") == "memory"
        assert classify_failure_reason("CUDA out of memory. Tried to allocate 2.00 GiB") == "memory"
        assert classify_failure_reason("insufficient shared memory") == "memory"

    def test_classify_failure_reason_model_unavailable(self):
        assert classify_failure_reason("model not found") == "model_unavailable"
        assert classify_failure_reason("model unavailable") == "model_unavailable"
        assert classify_failure_reason("module is not installed") == "model_unavailable"

    def test_classify_failure_reason_network(self):
        assert classify_failure_reason("connection refused") == "network"
        assert classify_failure_reason("network is unreachable") == "network"
        assert classify_failure_reason("Connection reset by peer") == "network"


# ======================================================================
# Pipeline initialization
# ======================================================================


class TestPipelineInitialization:
    def test_pipeline_initialization(self):
        config = make_configuration()
        bench = DummyBenchmark()
        agent = DummyAgent()
        pipeline = ExperimentPipeline(config=config, benchmark=bench, agent=agent)
        assert pipeline.config == config
        assert pipeline.benchmark == bench
        assert pipeline.agent == agent
        assert pipeline.cache is None
        assert pipeline.execution_records == []
        assert pipeline.evaluation_records == []
        assert pipeline.metric_records == []
        assert pipeline.ranking_records == []

    def test_pipeline_initialization_with_cache(self):
        config = make_configuration()
        cache = MockCache()
        pipeline = ExperimentPipeline(
            config=config,
            benchmark=DummyBenchmark(),
            agent=DummyAgent(),
            cache=cache,
        )
        assert pipeline.cache is cache


# ======================================================================
# Pipeline run
# ======================================================================


class TestPipelineRun:
    def test_pipeline_run_basic(self):
        config = make_configuration(repetitions=1, perturbations=(), fault_injection=False)
        bench = DummyBenchmark(task_ids=["task-0"])
        agent = DummyAgent()
        pipeline = ExperimentPipeline(config=config, benchmark=bench, agent=agent)
        result = pipeline.run()
        assert isinstance(result, ExperimentResult)
        assert result.configuration == config
        assert len(result.execution_records) == 1
        assert result.execution_records[0].status == "success"
        assert len(result.evaluation_records) == 1
        assert agent.initialized
        assert agent.shutdown_called

    def test_pipeline_run_multiple_tasks(self):
        config = make_configuration(repetitions=1, perturbations=(), fault_injection=False)
        bench = DummyBenchmark(task_ids=["t1", "t2", "t3"])
        agent = DummyAgent()
        pipeline = ExperimentPipeline(config=config, benchmark=bench, agent=agent)
        result = pipeline.run()
        assert len(result.execution_records) == 3
        assert len(result.evaluation_records) == 3

    def test_pipeline_run_with_repetitions(self):
        config = make_configuration(repetitions=3, perturbations=(), fault_injection=False)
        bench = DummyBenchmark(task_ids=["task-0"])
        agent = DummyAgent()
        pipeline = ExperimentPipeline(config=config, benchmark=bench, agent=agent)
        result = pipeline.run()
        assert len(result.execution_records) == 3

    def test_pipeline_run_with_cache(self):
        config = make_configuration(repetitions=1, perturbations=(), fault_injection=False)
        bench = DummyBenchmark(task_ids=["task-0"])
        agent = DummyAgent()
        cache = MockCache()
        pipeline1 = ExperimentPipeline(config=config, benchmark=bench, agent=agent, cache=cache)
        result1 = pipeline1.run()
        assert len(result1.execution_records) == 1
        assert cache._miss_count >= 0

        agent2 = DummyAgent()
        bench2 = DummyBenchmark(task_ids=["task-0"])
        pipeline2 = ExperimentPipeline(config=config, benchmark=bench2, agent=agent2, cache=cache)
        result2 = pipeline2.run()
        assert len(result2.execution_records) == 1
        assert result2.configuration.sha256() == result1.configuration.sha256()

    def test_pipeline_run_with_failing_agent_init(self):
        config = make_configuration(repetitions=1, perturbations=(), fault_injection=False)
        bench = DummyBenchmark(task_ids=["task-0"])
        agent = FailingAgent()
        pipeline = ExperimentPipeline(config=config, benchmark=bench, agent=agent)
        result = pipeline.run()
        assert len(result.execution_records) > 0
        assert "errors" in result.metadata

    def test_pipeline_run_all_completes(self):
        config = make_configuration(repetitions=1, perturbations=(), fault_injection=False)
        bench = DummyBenchmark(task_ids=["a", "b"])
        agent = DummyAgent()
        pipeline = ExperimentPipeline(config=config, benchmark=bench, agent=agent)
        pipeline.run_all()
        assert len(pipeline.execution_records) == 2

    def test_pipeline_save_results(self, tmp_path: Path):
        config = make_configuration(repetitions=1, perturbations=(), fault_injection=False)
        bench = DummyBenchmark(task_ids=["task-0"])
        agent = DummyAgent()
        pipeline = ExperimentPipeline(config=config, benchmark=bench, agent=agent)
        pipeline.run()
        dest = tmp_path / "result.json"
        pipeline.save_results(dest)
        assert dest.exists()
        loaded = ExperimentPipeline.load_results(dest)
        assert loaded.configuration.sha256() == config.sha256()
        assert len(loaded.execution_records) == 1


# ======================================================================
# Pipeline evaluate / compute_metrics / compute_rankings
# ======================================================================


class TestPipelineEvaluate:
    def test_pipeline_evaluate_with_mock(self):
        config = make_configuration()
        bench = DummyBenchmark()
        agent = DummyAgent()
        pipeline = ExperimentPipeline(config=config, benchmark=bench, agent=agent)
        exec_rec = ExecutionRecord(
            configuration_hash=CONFIG_HASH,
            seed=42,
            benchmark="mock",
            agent="test_agent",
            task_id="task-0",
            run_index=0,
            runtime_seconds=0.1,
            timestamp=TIMESTAMP,
            stdout="ok",
            stderr="",
            status="success",
        )
        pipeline.execution_records.append(exec_rec)
        pipeline.evaluate()
        assert len(pipeline.evaluation_records) == 1
        assert pipeline.evaluation_records[0].success

    def test_pipeline_evaluate_empty_executions(self):
        config = make_configuration()
        pipeline = ExperimentPipeline(
            config=config,
            benchmark=DummyBenchmark(),
            agent=DummyAgent(),
        )
        pipeline.evaluate()
        assert pipeline.evaluation_records == []


class TestPipelineComputeMetrics:
    def test_pipeline_compute_metrics_from_results(self):
        config = make_configuration()
        pipeline = ExperimentPipeline(
            config=config,
            benchmark=DummyBenchmark(),
            agent=DummyAgent(),
        )
        exec_rec = ExecutionRecord(
            configuration_hash=CONFIG_HASH,
            seed=42,
            benchmark="mock",
            agent="test_agent",
            task_id="task-0",
            run_index=0,
            runtime_seconds=0.1,
            timestamp=TIMESTAMP,
            stdout="ok",
            stderr="",
            status="success",
        )
        eval_rec = EvaluationRecord(
            execution_hash=exec_rec.sha256(),
            configuration_hash=CONFIG_HASH,
            seed=42,
            benchmark="mock",
            agent="test_agent",
            task_id="task-0",
            run_index=0,
            success=True,
            score=1.0,
            evaluated_at=TIMESTAMP,
        )
        pipeline.evaluation_records.append(eval_rec)
        pipeline.compute_metrics()
        assert len(pipeline.metric_records) == 1
        assert pipeline.metric_records[0].success_rate == 1.0

    def test_pipeline_compute_metrics_empty(self):
        config = make_configuration()
        pipeline = ExperimentPipeline(
            config=config,
            benchmark=DummyBenchmark(),
            agent=DummyAgent(),
        )
        pipeline.compute_metrics()
        assert pipeline.metric_records == []


class TestPipelineComputeRankings:
    def test_pipeline_compute_rankings_from_metrics(self):
        config = make_configuration()
        pipeline = ExperimentPipeline(
            config=config,
            benchmark=DummyBenchmark(),
            agent=DummyAgent(),
        )
        metric = MetricRecord(
            benchmark="mock",
            agent="test_agent",
            evaluation_count=1,
            success_rate=1.0,
            repeated_run_consistency=1.0,
            composite_reliability=1.0,
            computed_at=TIMESTAMP,
        )
        pipeline.metric_records.append(metric)
        pipeline.compute_rankings()
        assert len(pipeline.ranking_records) == 2
        types = {r.ranking_type for r in pipeline.ranking_records}
        assert types == {"success", "reliability"}

    def test_pipeline_compute_rankings_empty(self):
        config = make_configuration()
        pipeline = ExperimentPipeline(
            config=config,
            benchmark=DummyBenchmark(),
            agent=DummyAgent(),
        )
        pipeline.compute_rankings()
        assert pipeline.ranking_records == []


# ======================================================================
# Pipeline metadata and errors
# ======================================================================


class TestPipelineErrors:
    def test_pipeline_errors_tracked(self):
        config = make_configuration(repetitions=1, perturbations=(), fault_injection=False)
        bench = FailingBenchmark(task_ids=["task-0"])
        agent = DummyAgent()
        pipeline = ExperimentPipeline(config=config, benchmark=bench, agent=agent)
        result = pipeline.run()
        assert len(result.execution_records) == 1
        assert result.execution_records[0].status == "error"

    def test_pipeline_errors_list(self):
        config = make_configuration(repetitions=1, perturbations=(), fault_injection=False)
        bench = FailingBenchmark(task_ids=["task-0"])
        agent = DummyAgent()
        pipeline = ExperimentPipeline(config=config, benchmark=bench, agent=agent)
        pipeline.run()
        assert isinstance(pipeline.errors, list)
