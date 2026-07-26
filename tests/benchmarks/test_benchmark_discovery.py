import pytest

from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry


def _all_new_benchmarks() -> list[str]:
    return [
        "MMLU-Pro",
        "ARC-Challenge",
        "BBH",
        "DROP",
        "BoolQ",
        "CommonsenseQA",
        "OpenBookQA",
        "GPQA",
        "TriviaQA",
        "NaturalQuestions",
        "HotpotQA",
        "IFEval",
        "BIG-Bench-Lite",
        "ArenaHard",
        "LiveCodeBench",
    ]


def test_registry_contains_all_new_benchmarks():
    registered = BenchmarkRegistry.list()
    for name in _all_new_benchmarks():
        assert name in registered, f"{name} not found in registry"


def test_registry_all_15_new_exist():
    for name in _all_new_benchmarks():
        assert BenchmarkRegistry.exists(name)


def test_registry_get_returns_adapter():
    for name in _all_new_benchmarks():
        cls = BenchmarkRegistry.get(name)
        from llm_reliability.benchmarks.adapters.base_adapter import \
            BaseBenchmarkAdapter

        assert issubclass(cls, BaseBenchmarkAdapter)


@pytest.mark.parametrize(
    "name",
    [
        "MMLU-Pro",
        "ARC-Challenge",
        "BBH",
        "DROP",
    ],
)
def test_benchmark_registration_roundtrip(name):
    cls = BenchmarkRegistry.get(name)
    BenchmarkRegistry.unregister(name)
    assert not BenchmarkRegistry.exists(name)
    BenchmarkRegistry.register(name, cls)
    assert BenchmarkRegistry.exists(name)
    assert BenchmarkRegistry.get(name) is cls


def test_registry_includes_original_benchmarks():
    originals = [
        "AgentBoard",
        "ARC",
        "GAIA",
        "GSM8K",
        "HellaSwag",
        "HumanEval",
        "MBPP",
        "MMLU",
        "PIQA",
        "ReliabilityBench",
        "SWEBenchLite",
        "TruthfulQA",
        "Winogrande",
    ]
    registered = BenchmarkRegistry.list()
    for name in originals:
        assert name in registered


def test_registry_combined_count():
    registered = BenchmarkRegistry.list()
    assert len(registered) >= 28


def test_unknown_benchmark_not_found():
    assert not BenchmarkRegistry.exists("NonExistentBenchmark")
    with pytest.raises(ValueError, match="not found in registry"):
        BenchmarkRegistry.get("NonExistentBenchmark")


def test_duplicate_registration_raises():
    cls = BenchmarkRegistry.get("MMLU-Pro")
    with pytest.raises(ValueError, match="already registered"):
        BenchmarkRegistry.register("MMLU-Pro", cls)


def test_nonexistent_unregister_raises():
    with pytest.raises(ValueError, match="is not registered"):
        BenchmarkRegistry.unregister("TotallyFakeBenchmark")
