import pytest

from llm_reliability.benchmarks.adapters import (
    ArenaHardAdapter,
    BigBenchLiteAdapter,
    IFEvalAdapter,
    LiveCodeBenchAdapter,
)
from llm_reliability.configs.config import Configuration
from llm_reliability.interfaces.agent import Agent
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord


class _MockAgent:
    def run(self, task):
        return "A"

    def initialize(self):
        pass

    def reset(self):
        pass

    def shutdown(self):
        pass

    def metadata(self):
        return {}


class _QualityMockAgent:
    def __init__(self, output):
        self._output = output

    def run(self, task):
        return self._output

    def initialize(self):
        pass

    def reset(self):
        pass

    def shutdown(self):
        pass

    def metadata(self):
        return {}


def _make_config(benchmark_name: str) -> Configuration:
    return Configuration(
        experiment_name="test",
        benchmark=benchmark_name,
        agent="mock",
        llm="mock",
        prompt_version="v1",
        dataset_version="1.0",
        seed=42,
        repetitions=1,
        metadata={"dataset_path": "/nonexistent/path.json"},
    )


def test_ifeval_adapter_load():
    config = _make_config("IFEval")
    adapter = IFEvalAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) == 5


def test_ifeval_adapter_run_evaluate():
    config = _make_config("IFEval")
    adapter = IFEvalAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    agent = _MockAgent()
    for task_id in tasks:
        task = adapter.get_task(task_id)
        exec_record = adapter.run(agent, task)
        assert isinstance(exec_record, ExecutionRecord)
        assert exec_record.status == "success"
        eval_record = adapter.evaluate(exec_record)
        assert isinstance(eval_record, EvaluationRecord)


def test_ifeval_satisfies_instruction():
    from llm_reliability.benchmarks.adapters.ifeval_adapter import satisfies_instruction

    assert satisfies_instruction("because it is good", "keywords")
    assert not satisfies_instruction("nothing special", "keywords")
    assert satisfies_instruction('{"key": "value"}', "json")
    assert not satisfies_instruction("plain text", "json")


def test_arena_hard_adapter():
    config = _make_config("ArenaHard")
    adapter = ArenaHardAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) == 5
    agent = _MockAgent()
    for task_id in tasks:
        task = adapter.get_task(task_id)
        exec_record = adapter.run(agent, task)
        assert exec_record.status == "success"
        eval_record = adapter.evaluate(exec_record)
        assert isinstance(eval_record, EvaluationRecord)


def test_arena_hard_quality_check():
    from llm_reliability.benchmarks.adapters.arena_hard_adapter import (
        check_quality,
        extract_keywords,
    )

    ref = "Recursion is a programming technique where a function calls itself"
    assert check_quality(ref, ref)
    assert not check_quality("", ref)
    keywords = extract_keywords(ref)
    assert isinstance(keywords, set)
    assert len(keywords) > 0


def test_arena_hard_exact_match():
    config = _make_config("ArenaHard")
    adapter = ArenaHardAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    task = adapter.get_task(tasks[0])
    ref = task["reference_answer"]
    agent = _QualityMockAgent(ref)
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    assert eval_record.success
    assert eval_record.score == 1.0


def test_livecodebench_adapter():
    config = _make_config("LiveCodeBench")
    adapter = LiveCodeBenchAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) == 5
    agent = _MockAgent()
    for task_id in tasks:
        task = adapter.get_task(task_id)
        exec_record = adapter.run(agent, task)
        assert exec_record.status == "success"
        eval_record = adapter.evaluate(exec_record)
        assert isinstance(eval_record, EvaluationRecord)


def test_livecodebench_code_extraction():
    from llm_reliability.benchmarks.adapters.livecodebench_adapter import (
        extract_code,
        has_syntactic_structure,
    )

    code = extract_code("```python\ndef foo(): pass\n```")
    assert "def foo(): pass" in code
    assert extract_code("no code") == "no code"
    assert extract_code("") == ""
    assert has_syntactic_structure("def add(a, b): return a + b")
    assert not has_syntactic_structure("")


def test_livecodebench_valid_code_passes():
    config = _make_config("LiveCodeBench")
    adapter = LiveCodeBenchAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    task = adapter.get_task(tasks[0])
    agent = _QualityMockAgent("def add(a, b): return a + b")
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    assert eval_record.success
    assert eval_record.score == 1.0


def test_bigbench_lite_adapter():
    config = _make_config("BIG-Bench-Lite")
    adapter = BigBenchLiteAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) == 5


def test_bigbench_lite_mc_extraction():
    from llm_reliability.benchmarks.adapters.bigbench_lite_adapter import (
        check_free_form,
        extract_mc_answer,
    )

    assert extract_mc_answer("The answer is B") == "B"
    assert extract_mc_answer("A") == "A"
    assert extract_mc_answer("") == ""
    assert check_free_form("answer is 4", ["4"])
    assert check_free_form("oxygen is a gas", ["oxygen", "nitrogen"])


def test_bigbench_lite_run_and_evaluate():
    config = _make_config("BIG-Bench-Lite")
    adapter = BigBenchLiteAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    agent = _MockAgent()
    for task_id in tasks:
        task = adapter.get_task(task_id)
        exec_record = adapter.run(agent, task)
        assert exec_record.status == "success"
        assert exec_record.benchmark == "BIG-Bench-Lite"
        eval_record = adapter.evaluate(exec_record)
        assert isinstance(eval_record, EvaluationRecord)


def test_ifeval_adapter_metadata():
    config = _make_config("IFEval")
    adapter = IFEvalAdapter(config)
    meta = adapter.metadata()
    assert meta["name"] == "IFEval"
    adapter.load()
    assert adapter.metadata()["task_count"] == 5


def test_arena_hard_adapter_metadata():
    config = _make_config("ArenaHard")
    adapter = ArenaHardAdapter(config)
    meta = adapter.metadata()
    assert meta["name"] == "ArenaHard"
    adapter.load()
    assert adapter.metadata()["task_count"] == 5


def test_livecodebench_adapter_metadata():
    config = _make_config("LiveCodeBench")
    adapter = LiveCodeBenchAdapter(config)
    meta = adapter.metadata()
    assert meta["name"] == "LiveCodeBench"
    adapter.load()
    assert adapter.metadata()["task_count"] == 5
