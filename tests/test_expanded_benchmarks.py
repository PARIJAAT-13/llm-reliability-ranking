"""
Unit tests for expanded benchmark adapters (MMLU, HellaSwag, HumanEval, MBPP, TruthfulQA, GSM8K, ARC, Winogrande, PIQA).
"""

import pytest

from llm_reliability.benchmarks.adapters import (
    ARCAdapter,
    BenchmarkRegistry,
    GSM8KAdapter,
    HellaSwagAdapter,
    HumanEvalAdapter,
    MBPPAdapter,
    MMLUAdapter,
    PIQAAdapter,
    TruthfulQAAdapter,
    WinograndeAdapter,
)
from llm_reliability.configs.config import Configuration


def make_bench_config(benchmark_name: str) -> Configuration:
    return Configuration(
        experiment_name="test_expanded",
        agent="mock",
        benchmark=benchmark_name,
        llm="mock",
        prompt_version="v1",
        dataset_version="1.0",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": "non_existent.json"},
    )


def test_registry_contains_all_12_benchmarks():
    expected = [
        "AgentBoard",
        "ARC",
        "GAIA",
        "GSM8K",
        "HellaSwag",
        "HumanEval",
        "MBPP",
        "MMLU",
        "PIQA",
        "SWEBenchLite",
        "TruthfulQA",
        "Winogrande",
    ]
    for exp in expected:
        assert BenchmarkRegistry.exists(exp), f"Benchmark {exp} not registered in registry."


def test_mmlu_adapter():
    config = make_bench_config("MMLU")
    adapter = MMLUAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) > 0
    task = adapter.get_task(tasks[0])
    assert "prompt" in task


def test_hellaswag_adapter():
    config = make_bench_config("HellaSwag")
    adapter = HellaSwagAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) > 0


def test_humaneval_adapter():
    config = make_bench_config("HumanEval")
    adapter = HumanEvalAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) > 0


def test_mbpp_adapter():
    config = make_bench_config("MBPP")
    adapter = MBPPAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) > 0


def test_truthfulqa_adapter():
    config = make_bench_config("TruthfulQA")
    adapter = TruthfulQAAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) > 0


def test_gsm8k_adapter():
    config = make_bench_config("GSM8K")
    adapter = GSM8KAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) > 0


def test_arc_adapter():
    config = make_bench_config("ARC")
    adapter = ARCAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) > 0


def test_winogrande_adapter():
    config = make_bench_config("Winogrande")
    adapter = WinograndeAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) > 0


def test_piqa_adapter():
    config = make_bench_config("PIQA")
    adapter = PIQAAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) > 0
