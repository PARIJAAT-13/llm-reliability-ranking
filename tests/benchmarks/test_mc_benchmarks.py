import pytest

from llm_reliability.benchmarks.adapters import (ARCChallengeAdapter,
                                                 BBHAdapter, BoolQAdapter,
                                                 CommonsenseQAAdapter,
                                                 GPQAAdapter, MMLUProAdapter,
                                                 OpenBookQAAdapter)
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


@pytest.mark.parametrize(
    "adapter_cls,bench_name,expected_success",
    [
        (MMLUProAdapter, "MMLU-Pro", True),
        (ARCChallengeAdapter, "ARC-Challenge", True),
        (OpenBookQAAdapter, "OpenBookQA", True),
    ],
)
def test_mc_adapter_success(adapter_cls, bench_name, expected_success):
    config = _make_config(bench_name)
    adapter = adapter_cls(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) == 5
    agent = _MockAgent()
    for task_id in tasks:
        task = adapter.get_task(task_id)
        assert "prompt" in task
        exec_record = adapter.run(agent, task)
        assert isinstance(exec_record, ExecutionRecord)
        assert exec_record.status == "success"
        eval_record = adapter.evaluate(exec_record)
        assert isinstance(eval_record, EvaluationRecord)
        assert eval_record.success == expected_success
        assert eval_record.score == (1.0 if expected_success else 0.0)


@pytest.mark.parametrize(
    "adapter_cls,bench_name,expected_success",
    [
        (BBHAdapter, "BBH", False),
        (BoolQAdapter, "BoolQ", False),
        (CommonsenseQAAdapter, "CommonsenseQA", False),
        (GPQAAdapter, "GPQA", False),
    ],
)
def test_mc_adapter_mismatch(adapter_cls, bench_name, expected_success):
    config = _make_config(bench_name)
    adapter = adapter_cls(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) == 5
    agent = _MockAgent()
    for task_id in tasks:
        task = adapter.get_task(task_id)
        exec_record = adapter.run(agent, task)
        assert isinstance(exec_record, ExecutionRecord)
        assert exec_record.status == "success"
        assert exec_record.agent_output == "A"
        eval_record = adapter.evaluate(exec_record)
        assert isinstance(eval_record, EvaluationRecord)
        assert eval_record.success == expected_success
        assert eval_record.score == (1.0 if expected_success else 0.0)


def test_mmlu_pro_adapter_metadata():
    config = _make_config("MMLU-Pro")
    adapter = MMLUProAdapter(config)
    meta = adapter.metadata()
    assert meta["name"] == "MMLU-Pro"
    assert meta["version"] == "1.0"
    assert meta["deterministic"]
    adapter.load()
    meta2 = adapter.metadata()
    assert meta2["task_count"] == 5


def test_arc_challenge_adapter_metadata():
    config = _make_config("ARC-Challenge")
    adapter = ARCChallengeAdapter(config)
    meta = adapter.metadata()
    assert meta["name"] == "ARC-Challenge"
    meta = adapter.collect_logs()
    assert isinstance(meta["logs"], list)
    adapter.load()
    logs = adapter.collect_logs()
    assert len(logs["logs"]) == 1
    assert logs["logs"][0]["event"] == "load"


def test_bbh_adapter_evaluate_error():
    config = _make_config("BBH")
    adapter = BBHAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    _ = adapter.get_task(tasks[0])
    from llm_reliability.records.execution import ExecutionRecord

    error_exec = ExecutionRecord(
        configuration_hash="a" * 64,
        seed=42,
        benchmark="BBH",
        agent="mock",
        task_id=tasks[0],
        run_index=0,
        runtime_seconds=1.0,
        timestamp="2026-01-01T00:00:00",
        stdout="",
        stderr="error",
        status="error",
        error="fail",
        agent_output=None,
        software_versions={},
        environment_metadata={},
    )
    eval_record = adapter.evaluate(error_exec)
    assert not eval_record.success
    assert eval_record.score == 0.0


def test_boolq_adapter_extraction():
    from llm_reliability.benchmarks.adapters.boolq_adapter import \
        extract_boolq_answer

    assert extract_boolq_answer("True") == "TRUE"
    assert extract_boolq_answer("yes") == "TRUE"
    assert extract_boolq_answer("False") == "FALSE"
    assert extract_boolq_answer("no") == "FALSE"
    assert extract_boolq_answer("") == ""
    assert extract_boolq_answer("maybe") == ""


def test_commonsenseqa_adapter_validate_config():
    cfg = Configuration(
        experiment_name="test",
        benchmark="CommonsenseQA",
        agent="mock",
        llm="mock",
        prompt_version="v1",
        dataset_version="1.0",
        seed=42,
        repetitions=1,
        metadata={},
    )
    with pytest.raises(ValueError, match="dataset_path"):
        CommonsenseQAAdapter(cfg)
