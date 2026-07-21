"""Shared test helpers for the reliability metrics test suite."""

from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord


def _make_execution(
    task_id: str = "t1",
    run_index: int = 0,
    seed: int = 42,
    perturbation: str | None = None,
    fault_injected: bool = False,
) -> ExecutionRecord:
    return ExecutionRecord(
        configuration_hash="a" * 64,
        seed=seed,
        benchmark="mock",
        agent="mock",
        task_id=task_id,
        run_index=run_index,
        runtime_seconds=1.0,
        timestamp="2026-01-01T00:00:00+00:00",
        stdout="",
        stderr="",
        status="success",
        error=None,
        agent_output="ans",
        software_versions={},
        environment_metadata={},
        perturbation=perturbation,
        fault_injected=fault_injected,
    )


def make_eval(
    task_id: str = "t1",
    success: bool = True,
    score: float = 1.0,
    run_index: int = 0,
    perturbation: str | None = None,
    fault_injected: bool = False,
    seed: int = 42,
) -> EvaluationRecord:
    """Factory for EvaluationRecord instances used across test modules."""
    exec_rec = _make_execution(
        task_id=task_id,
        run_index=run_index,
        seed=seed,
        perturbation=perturbation,
        fault_injected=fault_injected,
    )
    return EvaluationRecord.from_execution(
        execution=exec_rec,
        success=success,
        score=score,
        evaluated_at="2026-01-01T01:00:00+00:00",
    )
