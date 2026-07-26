"""Full pipeline integration tests — end-to-end experiment execution with all components."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
from llm_reliability.cache import ExperimentCache, FileSystemCacheBackend
from llm_reliability.configs.config import Configuration
from llm_reliability.experiments.experiment_models import (AgentSpec,
                                                           BenchmarkSpec,
                                                           ExperimentSpec,
                                                           ExperimentState)
from llm_reliability.experiments.experiment_runner import ExperimentRunner
from llm_reliability.experiments.scheduler import RunDescriptor, Scheduler
from llm_reliability.interfaces.agent import Agent
from llm_reliability.pipeline.experiment_pipeline import (ExperimentPipeline,
                                                          ExperimentResult)
from llm_reliability.runtime.batching import BatchExecutor
from llm_reliability.runtime.cost_accounting import (CostCalculator, CostEntry,
                                                     TokenAccount, TokenUsage)


class DummyAgent(Agent):
    def initialize(self):
        pass

    def reset(self):
        pass

    def run(self, task: dict):
        return task.get("expected_answer", "")

    def shutdown(self):
        pass

    def metadata(self):
        return {"name": "dummy"}


class TestPipelineEndToEnd:
    def test_basic_pipeline_run(self):
        config = Configuration(
            experiment_name="integration_test",
            benchmark="mock",
            agent="dummy",
            llm="test-llm",
            prompt_version="v1",
            dataset_version="v1",
            seed=42,
            repetitions=2,
        )
        benchmark = MockBenchmark(seed=42)
        agent = DummyAgent()
        pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=agent)
        result = pipeline.run()
        assert isinstance(result, ExperimentResult)
        assert len(result.execution_records) > 0
        assert len(result.evaluation_records) > 0
        assert len(result.metric_records) > 0
        assert len(result.ranking_records) > 0

    def test_execution_records_generated(self):
        config = Configuration(
            experiment_name="exec_test",
            benchmark="mock",
            agent="dummy",
            llm="test-llm",
            prompt_version="v1",
            dataset_version="v1",
            seed=42,
            repetitions=1,
        )
        benchmark = MockBenchmark(seed=42)
        agent = DummyAgent()
        pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=agent)
        result = pipeline.run()
        assert len(result.execution_records) == 10
        for rec in result.execution_records:
            assert rec.task_id.startswith("mock-task-")
            assert rec.status == "success"
            assert rec.runtime_seconds > 0

    def test_evaluation_records_generated(self):
        config = Configuration(
            experiment_name="eval_test",
            benchmark="mock",
            agent="dummy",
            llm="test-llm",
            prompt_version="v1",
            dataset_version="v1",
            seed=42,
            repetitions=1,
        )
        benchmark = MockBenchmark(seed=42)
        agent = DummyAgent()
        pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=agent)
        result = pipeline.run()
        assert len(result.evaluation_records) == 10
        for rec in result.evaluation_records:
            assert rec.success is True
            assert rec.score == 1.0

    def test_metric_records_generated(self):
        config = Configuration(
            experiment_name="metric_test",
            benchmark="mock",
            agent="dummy",
            llm="test-llm",
            prompt_version="v1",
            dataset_version="v1",
            seed=42,
            repetitions=3,
        )
        benchmark = MockBenchmark(seed=42)
        agent = DummyAgent()
        pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=agent)
        result = pipeline.run()
        assert len(result.metric_records) >= 1
        for rec in result.metric_records:
            assert rec.success_rate == 1.0
            assert rec.evaluation_count > 0

    def test_ranking_records_generated(self):
        config = Configuration(
            experiment_name="rank_test",
            benchmark="mock",
            agent="dummy",
            llm="test-llm",
            prompt_version="v1",
            dataset_version="v1",
            seed=42,
            repetitions=2,
        )
        benchmark = MockBenchmark(seed=42)
        agent = DummyAgent()
        pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=agent)
        result = pipeline.run()
        assert len(result.ranking_records) >= 1

    def test_deterministic_output(self):
        config = Configuration(
            experiment_name="det_test",
            benchmark="mock",
            agent="dummy",
            llm="test-llm",
            prompt_version="v1",
            dataset_version="v1",
            seed=42,
            repetitions=2,
        )
        agent = DummyAgent()
        benchmark1 = MockBenchmark(seed=42)
        pipeline1 = ExperimentPipeline(config=config, benchmark=benchmark1, agent=agent)
        result1 = pipeline1.run()

        benchmark2 = MockBenchmark(seed=42)
        pipeline2 = ExperimentPipeline(config=config, benchmark=benchmark2, agent=agent)
        result2 = pipeline2.run()

        for e1, e2 in zip(result1.execution_records, result2.execution_records):
            assert e1.sha256() == e2.sha256()


class TestPipelineWithCache:
    def test_pipeline_with_cache_hit(self):
        config = Configuration(
            experiment_name="cache_test",
            benchmark="mock",
            agent="dummy",
            llm="test-llm",
            prompt_version="v1",
            dataset_version="v1",
            seed=42,
            repetitions=1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileSystemCacheBackend(cache_dir=Path(tmpdir))
            cache = ExperimentCache(backend=backend, enabled=True)
            agent = DummyAgent()

            benchmark1 = MockBenchmark(seed=42)
            pipeline1 = ExperimentPipeline(
                config=config, benchmark=benchmark1, agent=agent, cache=cache
            )
            result1 = pipeline1.run()

            benchmark2 = MockBenchmark(seed=42)
            pipeline2 = ExperimentPipeline(
                config=config, benchmark=benchmark2, agent=agent, cache=cache
            )
            result2 = pipeline2.run()

            assert len(result1.execution_records) == len(result2.execution_records)

    def test_pipeline_cache_miss_then_hit(self):
        config = Configuration(
            experiment_name="miss_then_hit",
            benchmark="mock",
            agent="dummy",
            llm="test-llm",
            prompt_version="v1",
            dataset_version="v1",
            seed=42,
            repetitions=1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileSystemCacheBackend(cache_dir=Path(tmpdir))
            cache = ExperimentCache(backend=backend, enabled=True)
            agent = DummyAgent()

            key = cache.generate_key(config)
            assert cache.get(key) is None

            benchmark = MockBenchmark(seed=42)
            pipeline = ExperimentPipeline(
                config=config, benchmark=benchmark, agent=agent, cache=cache
            )
            pipeline.run()

            assert cache.exists(key)
            cached = cache.get(key)
            assert cached is not None

    def test_pipeline_with_disabled_cache(self):
        config = Configuration(
            experiment_name="no_cache",
            benchmark="mock",
            agent="dummy",
            llm="test-llm",
            prompt_version="v1",
            dataset_version="v1",
            seed=42,
            repetitions=1,
        )
        agent = DummyAgent()
        cache = ExperimentCache(enabled=False)
        benchmark = MockBenchmark(seed=42)
        pipeline = ExperimentPipeline(config=config, benchmark=benchmark, agent=agent, cache=cache)
        result = pipeline.run()
        assert len(result.execution_records) == 10


class TestExperimentRunnerIntegration:
    def _make_benchmark_factory(self):
        def benchmark_factory(name, config):
            return MockBenchmark(config=config)

        return benchmark_factory

    def test_experiment_runner_basic(self):
        spec = ExperimentSpec(
            experiment_name="runner_test",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="test.json")],
            agents=[AgentSpec(name="mock")],
            seeds=[42],
            repetitions=2,
            llm="mock",
            prompt_version="1",
            dataset_version="1",
            output_dir=tempfile.mkdtemp(),
        )

        def agent_factory(aspec, config):
            return DummyAgent()

        runner = ExperimentRunner(
            spec, agent_factory=agent_factory, benchmark_factory=self._make_benchmark_factory()
        )
        status = runner.run()
        assert status.state in (ExperimentState.COMPLETED, ExperimentState.FAILED)

    def test_experiment_runner_generates_all_record_types(self):
        import time as _time

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            spec = ExperimentSpec(
                experiment_name="all_records",
                benchmarks=[BenchmarkSpec(name="mock", dataset_path="test.json")],
                agents=[AgentSpec(name="mock")],
                seeds=[42],
                repetitions=2,
                llm="mock",
                prompt_version="1",
                dataset_version="1",
                output_dir=tmpdir,
            )

            def agent_factory(aspec, config):
                return DummyAgent()

            runner = ExperimentRunner(
                spec, agent_factory=agent_factory, benchmark_factory=self._make_benchmark_factory()
            )
            runner.run()

            assert len(runner.executions) > 0
            assert len(runner.evaluations) > 0

    def test_experiment_runner_stop(self):
        spec = ExperimentSpec(
            experiment_name="stop_test",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="test.json")],
            agents=[AgentSpec(name="mock")],
            seeds=[42],
            repetitions=1,
            llm="mock",
            prompt_version="1",
            dataset_version="1",
            output_dir=tempfile.mkdtemp(),
        )

        def agent_factory(aspec, config):
            return DummyAgent()

        runner = ExperimentRunner(
            spec, agent_factory=agent_factory, benchmark_factory=self._make_benchmark_factory()
        )
        runner.stop()
        status = runner.run()
        assert status.state == ExperimentState.PAUSED


class TestScheduler:
    def test_scheduler_total_runs(self):
        spec = ExperimentSpec(
            experiment_name="sched_test",
            benchmarks=[BenchmarkSpec(name="b1", dataset_path="d1.json")],
            agents=[AgentSpec(name="a1")],
            seeds=[42, 43],
            repetitions=3,
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        scheduler = Scheduler(spec)
        assert scheduler.total_runs() == 1 * 1 * 2 * 3

    def test_scheduler_run_queue_length(self):
        spec = ExperimentSpec(
            experiment_name="queue_test",
            benchmarks=[
                BenchmarkSpec(name="b1", dataset_path="d1.json"),
                BenchmarkSpec(name="b2", dataset_path="d2.json"),
            ],
            agents=[AgentSpec(name="a1"), AgentSpec(name="a2")],
            seeds=[42],
            repetitions=2,
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        scheduler = Scheduler(spec)
        queue = scheduler.build_run_queue()
        assert len(queue) == 2 * 2 * 1 * 2

    def test_run_descriptor_fields(self):
        spec = ExperimentSpec(
            experiment_name="desc_test",
            benchmarks=[BenchmarkSpec(name="mock", dataset_path="data.json")],
            agents=[AgentSpec(name="test_agent")],
            seeds=[42],
            repetitions=1,
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        scheduler = Scheduler(spec)
        queue = scheduler.build_run_queue()
        assert len(queue) == 1
        desc = queue[0]
        assert isinstance(desc, RunDescriptor)
        assert desc.benchmark_name == "mock"
        assert desc.agent_name == "test_agent"
        assert desc.base_seed == 42
        assert desc.run_index == 0
        assert desc.dataset_path == "data.json"
        assert desc.derived_seed >= 0

    def test_scheduler_with_multiple_benchmarks_and_agents(self):
        spec = ExperimentSpec(
            experiment_name="multi",
            benchmarks=[
                BenchmarkSpec(name="bench_a", dataset_path="a.json"),
                BenchmarkSpec(name="bench_b", dataset_path="b.json"),
                BenchmarkSpec(name="bench_c", dataset_path="c.json"),
            ],
            agents=[AgentSpec(name="agent_x"), AgentSpec(name="agent_y")],
            seeds=[1, 2],
            repetitions=2,
            llm="mock",
            prompt_version="1",
            dataset_version="1",
        )
        scheduler = Scheduler(spec)
        queue = scheduler.build_run_queue()
        assert len(queue) == 3 * 2 * 2 * 2


class _ExecutableDummy(Agent):
    def initialize(self):
        pass

    def reset(self):
        pass

    def run(self, task):
        return task.get("expected_answer", "")

    def shutdown(self):
        pass

    def metadata(self):
        return {"name": "exec_dummy"}

    def execute(self, task):
        return self.run(task)


class TestBatchExecutor:
    def test_batch_executor_basic(self):
        agent = _ExecutableDummy()
        executor = BatchExecutor(executor=agent, max_batch_size=4)
        tasks = [{"task_id": f"t{i}", "expected_answer": f"answer{i}"} for i in range(4)]
        result = executor.execute_batch(tasks)
        assert len(result.results) == 4
        assert result.batch_size == 4
        assert result.batch_duration_ms >= 0

    def test_batch_executor_with_errors(self):
        class ErrorExecAgent(Agent):
            def initialize(self):
                pass

            def reset(self):
                pass

            def run(self, task):
                raise ValueError("error")

            def shutdown(self):
                pass

            def metadata(self):
                return {"name": "error"}

            def execute(self, task):
                raise ValueError("error")

        agent = ErrorExecAgent()
        executor = BatchExecutor(executor=agent)
        tasks = [{"task_id": f"t{i}"} for i in range(3)]
        result = executor.execute_batch(tasks)
        assert len(result.errors) == 3
        assert all(r is None for r in result.results)

    def test_batch_executor_streaming(self):
        agent = _ExecutableDummy()
        executor = BatchExecutor(executor=agent)
        tasks = [
            {"task_id": "t1", "expected_answer": "a1"},
            {"task_id": "t2", "expected_answer": "a2"},
        ]
        result = executor.execute_batch(tasks)
        assert len(result.results) == 2
        assert result.results[0] == "a1"


class TestCostCalculator:
    def test_cost_calculator_estimate(self):
        cost = CostCalculator.estimate_cost("openai", "gpt-4o", 100, 50)
        assert cost > 0

    def test_cost_calculator_unknown_provider(self):
        cost = CostCalculator.estimate_cost("unknown_provider", "unknown_model", 100, 50)
        assert cost == 0

    def test_cost_calculator_unknown_model(self):
        cost = CostCalculator.estimate_cost("openai", "nonexistent-model", 100, 50)
        assert cost >= 0

    def test_cost_calculator_known_model_zero_tokens(self):
        cost = CostCalculator.estimate_cost("openai", "gpt-4o", 0, 0)
        assert cost == 0

    def test_record_usage(self):
        entry = CostCalculator.record_usage("openai", "gpt-4o", 100, 50, 150.0)
        assert isinstance(entry, CostEntry)
        assert entry.provider == "openai"
        assert entry.model == "gpt-4o"
        assert entry.input_tokens == 100
        assert entry.output_tokens == 50
        assert entry.latency_ms == 150.0
        assert entry.cost_usd > 0


class TestTokenAccount:
    def test_token_account_empty(self):
        account = TokenAccount()
        assert account.total_tokens == 0
        assert account.total_cost_usd == 0
        assert account.entry_count == 0

    def test_token_account_add_entry(self):
        account = TokenAccount()
        entry = CostCalculator.record_usage("openai", "gpt-4o", 100, 50, 150.0)
        account.add(entry)
        assert account.entry_count == 1
        assert account.total_input_tokens == 100
        assert account.total_output_tokens == 50
        assert account.total_tokens == 150

    def test_token_account_summary(self):
        account = TokenAccount()
        entry = CostCalculator.record_usage(
            "anthropic", "claude-3-5-sonnet-20241022", 200, 100, 300.0
        )
        account.add(entry)
        summary = account.summary()
        assert summary["total_tokens"] == 300
        assert summary["call_count"] == 1
        assert summary["total_cost_usd"] > 0


class TestExperimentSpec:
    def test_experiment_spec_duplicate_benchmarks_raises(self):
        with pytest.raises(ValueError, match="Duplicate benchmark"):
            ExperimentSpec(
                experiment_name="dup",
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

    def test_experiment_spec_negative_seed_raises(self):
        with pytest.raises(ValueError, match="Seeds must be non-negative"):
            ExperimentSpec(
                experiment_name="neg",
                benchmarks=[BenchmarkSpec(name="mock", dataset_path="a.json")],
                agents=[AgentSpec(name="mock")],
                seeds=[-1],
                llm="mock",
                prompt_version="1",
                dataset_version="1",
            )
