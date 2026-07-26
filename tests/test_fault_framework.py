"""Tests for the Fault Injection Framework."""

from typing import Any, Optional

import pytest

from llm_reliability.agents.mock_agent import MockAgent
from llm_reliability.benchmarks.mock_benchmark import MockBenchmark
from llm_reliability.configs.config import Configuration
from llm_reliability.metrics.fault_tolerance import compute_fault_tolerance
from llm_reliability.reliability.faults import (
    ArtificialTimeoutFaultStrategy,
    ContextTruncationFaultStrategy,
    FaultInjectionStrategy,
    FaultManager,
    FaultReport,
    FaultReportGenerator,
    FaultRunResult,
    InvalidModelResponseFaultStrategy,
    NetworkInterruptionFaultStrategy,
    TemporaryApiFailureFaultStrategy,
    ToolFailureFaultStrategy,
)


class FaultyInjectionStrategy(FaultInjectionStrategy):
    """Strategy that raises an unexpected internal exception during inject()."""

    @property
    def fault_name(self) -> str:
        return "faulty_internal"

    @property
    def injection_point(self) -> str:
        return "prompt"

    @property
    def description(self) -> str:
        return "Internal failure strategy."

    def inject(self, target: Any, seed: int | None = None, **kwargs: Any) -> Any:
        raise RuntimeError("Unexpected internal crash in strategy code.")

    def cleanup(self) -> None:
        pass


def test_artificial_timeout_strategy():
    strategy = ArtificialTimeoutFaultStrategy(delay_seconds=0.01, raise_timeout=True)
    assert strategy.fault_name == "artificial_timeout"
    assert strategy.injection_point == "agent_run"

    with pytest.raises(TimeoutError, match="Artificial execution timeout"):
        strategy.inject(None)


def test_temporary_api_failure_strategy():
    strategy = TemporaryApiFailureFaultStrategy(max_failures=1)
    assert strategy.fault_name == "temporary_api_failure"
    assert strategy.injection_point == "api_call"

    # First attempt raises error
    with pytest.raises(RuntimeError, match="503 Service Unavailable"):
        strategy.inject(None)

    # Second attempt succeeds
    res = strategy.inject("ok")
    assert res == "ok"

    strategy.cleanup()
    assert strategy.current_failures == 0


def test_invalid_model_response_strategy():
    strategy = InvalidModelResponseFaultStrategy(mode="empty")
    assert strategy.inject(None) == ""

    strategy_json = InvalidModelResponseFaultStrategy(mode="malformed_json")
    assert "{" in strategy_json.inject(None)

    strategy_type = InvalidModelResponseFaultStrategy(mode="unexpected_type")
    assert strategy_type.inject(None) == 12345


def test_tool_failure_strategy():
    strategy = ToolFailureFaultStrategy(tool_name="web_search")
    assert strategy.fault_name == "tool_failure"

    with pytest.raises(RuntimeError, match="unavailable"):
        strategy.inject(None)


def test_context_truncation_strategy():
    strategy = ContextTruncationFaultStrategy(truncation_ratio=0.5)
    assert strategy.fault_name == "context_truncation"

    task = {"task_id": "t1", "prompt": "1234567890"}
    truncated = strategy.inject(task)
    assert len(truncated["prompt"]) == 5


def test_network_interruption_strategy():
    strategy = NetworkInterruptionFaultStrategy()
    assert strategy.fault_name == "network_interruption"

    with pytest.raises(ConnectionResetError):
        strategy.inject(None)


def test_fault_manager_fault_tolerance(sample_task: dict[str, Any] | None = None):
    # Setup mock benchmark and agent
    cfg = Configuration(
        experiment_name="fault_test",
        benchmark="MockBenchmark",
        agent="MockAgent",
        llm="mock",
        prompt_version="v1",
        dataset_version="v1",
        seed=42,
        repetitions=1,
        fault_injection=True,
    )
    benchmark = MockBenchmark(config=cfg)
    benchmark.load()
    agent = MockAgent(config=cfg)
    task = benchmark.get_task("mock-task-0")

    # Include internal faulty strategy along with valid ones
    faulty = FaultyInjectionStrategy()
    valid_strat = ContextTruncationFaultStrategy(truncation_ratio=0.2)
    manager = FaultManager(config=cfg, strategies=[faulty, valid_strat], max_retries=1)

    result = manager.run_fault_injected_task(agent, benchmark, task)

    assert isinstance(result, FaultRunResult)
    # The faulty strategy should be disabled and logged without crashing framework
    assert "faulty_internal" in manager.disabled_strategies
    assert len(result.execution_records) >= 2
    assert result.baseline_execution is not None
    assert not result.baseline_execution.fault_injected

    for f_exec in result.faulted_executions:
        assert f_exec.fault_injected
        assert f_exec.task_id == task["task_id"]


def test_fault_report_generation():
    cfg = Configuration(
        experiment_name="report_test",
        benchmark="MockBenchmark",
        agent="MockAgent",
        llm="mock",
        prompt_version="v1",
        dataset_version="v1",
        seed=42,
        repetitions=1,
        fault_injection=True,
    )
    benchmark = MockBenchmark(config=cfg)
    benchmark.load()
    agent = MockAgent(config=cfg)
    task = benchmark.get_task("mock-task-0")

    manager = FaultManager(
        config=cfg,
        strategies=[
            TemporaryApiFailureFaultStrategy(max_failures=1),
            ContextTruncationFaultStrategy(truncation_ratio=0.1),
        ],
        max_retries=2,
    )
    result = manager.run_fault_injected_task(agent, benchmark, task)

    report = FaultReportGenerator.generate_report(result)
    assert isinstance(report, FaultReport)
    assert report.total_fault_attempts == 2
    assert 0.0 <= report.overall_recovery_rate <= 1.0

    md = report.to_markdown()
    assert "# Fault Tolerance Summary Report" in md
    assert "| Fault Type |" in md


def test_end_to_end_fault_tolerance_metric_compatibility():
    cfg = Configuration(
        experiment_name="metric_compat",
        benchmark="MockBenchmark",
        agent="MockAgent",
        llm="mock",
        prompt_version="v1",
        dataset_version="v1",
        seed=42,
        repetitions=1,
        fault_injection=True,
    )
    benchmark = MockBenchmark(config=cfg)
    benchmark.load()
    agent = MockAgent(config=cfg)
    task = benchmark.get_task("mock-task-0")

    manager = FaultManager(config=cfg, max_retries=1)
    res = manager.run_fault_injected_task(agent, benchmark, task)

    # Verify downstream compatibility with compute_fault_tolerance metric
    ft_score = compute_fault_tolerance(res.evaluation_records)
    assert 0.0 <= ft_score <= 1.0
