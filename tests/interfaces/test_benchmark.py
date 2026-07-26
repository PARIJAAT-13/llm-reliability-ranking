"""Tests for Benchmark interface (Artifact 2)."""

import inspect

import pytest

from llm_reliability.interfaces import Benchmark


def test_benchmark_is_abstract() -> None:
    with pytest.raises(TypeError):
        Benchmark()


def test_benchmark_defines_required_methods() -> None:
    required = {
        "load",
        "list_tasks",
        "get_task",
        "run",
        "evaluate",
        "collect_logs",
        "metadata",
    }
    method_names = {
        name for name, member in inspect.getmembers(Benchmark, predicate=inspect.isfunction)
    }
    assert required.issubset(method_names)
    for name in required:
        assert getattr(Benchmark, name).__isabstractmethod__


class _MinimalBenchmark(Benchmark):
    def load(self) -> None:
        return None

    def list_tasks(self) -> list[str]:
        return ["task-1"]

    def get_task(self, task_id: str) -> dict[str, str]:
        return {"id": task_id}

    def run(self, agent: object, task: dict[str, str]) -> object:
        return object()

    def evaluate(self, execution: object) -> object:
        return object()

    def collect_logs(self) -> dict[str, str]:
        return {}

    def metadata(self) -> dict[str, str]:
        return {"name": "minimal"}


def test_concrete_benchmark_can_be_instantiated() -> None:
    benchmark = _MinimalBenchmark()
    assert benchmark.metadata() == {"name": "minimal"}
