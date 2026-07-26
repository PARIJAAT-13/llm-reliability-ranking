import pytest

from llm_reliability.benchmarks.adapters import (DROPAdapter, HotpotQAAdapter,
                                                 NaturalQuestionsAdapter,
                                                 TriviaQAAdapter)
from llm_reliability.configs.config import Configuration
from llm_reliability.interfaces.agent import Agent
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord


class _SmartMockAgent:
    def __init__(self, fallback_answer="Paris"):
        self._fallback = fallback_answer

    def run(self, task):
        return task.get(
            "ground_truth_answer",
            task.get(
                "answer",
                (
                    task.get("aliases", [self._fallback])[0]
                    if isinstance(task.get("aliases"), list) and task.get("aliases")
                    else self._fallback
                ),
            ),
        )

    def initialize(self):
        pass

    def reset(self):
        pass

    def shutdown(self):
        pass

    def metadata(self):
        return {}


class _FixedMockAgent:
    def __init__(self, answer="Paris"):
        self._answer = answer

    def run(self, task):
        return self._answer

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


def test_triviaqa_adapter():
    config = _make_config("TriviaQA")
    adapter = TriviaQAAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) == 5
    agent = _SmartMockAgent()
    for task_id in tasks:
        task = adapter.get_task(task_id)
        exec_record = adapter.run(agent, task)
        assert exec_record.status == "success"
        eval_record = adapter.evaluate(exec_record)
        assert eval_record.success
        assert eval_record.score == 1.0


def test_triviaqa_evaluate_no_match():
    config = _make_config("TriviaQA")
    adapter = TriviaQAAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    exec_record = adapter.run(_FixedMockAgent("NonexistentAnswer"), adapter.get_task(tasks[0]))
    eval_record = adapter.evaluate(exec_record)
    assert not eval_record.success
    assert eval_record.score == 0.0


def test_triviaqa_aliases():
    from llm_reliability.benchmarks.adapters.triviaqa_adapter import \
        extract_triviaqa_answer

    assert extract_triviaqa_answer("Paris is great", ["Paris"])
    assert extract_triviaqa_answer("I live in paris", ["Paris"])
    assert not extract_triviaqa_answer("London", ["Paris"])
    assert not extract_triviaqa_answer("", ["Paris"])


def test_natural_questions_adapter():
    config = _make_config("NaturalQuestions")
    adapter = NaturalQuestionsAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) == 5
    task = adapter.get_task(tasks[0])
    assert task["ground_truth_answer"] == "George Orwell"
    agent = _FixedMockAgent("George Orwell")
    exec_record = adapter.run(agent, task)
    assert exec_record.status == "success"
    eval_record = adapter.evaluate(exec_record)
    assert eval_record.success
    assert eval_record.score == 1.0


def test_natural_questions_short_answer_match():
    config = _make_config("NaturalQuestions")
    adapter = NaturalQuestionsAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    task = adapter.get_task(tasks[0])
    agent = _FixedMockAgent("Orwell")
    exec_record = adapter.run(agent, task)
    eval_record = adapter.evaluate(exec_record)
    assert eval_record.success


def test_natural_questions_no_match():
    config = _make_config("NaturalQuestions")
    adapter = NaturalQuestionsAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    exec_record = adapter.run(_FixedMockAgent("Unknown"), adapter.get_task(tasks[0]))
    eval_record = adapter.evaluate(exec_record)
    assert not eval_record.success


def test_hotpotqa_adapter():
    config = _make_config("HotpotQA")
    adapter = HotpotQAAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) == 5
    agent = _FixedMockAgent("Paris")
    for task_id in tasks:
        exec_record = adapter.run(agent, adapter.get_task(task_id))
        eval_record = adapter.evaluate(exec_record)
        assert eval_record.success
        assert eval_record.score > 0


def test_hotpotqa_exact_match_and_f1():
    from llm_reliability.benchmarks.adapters.hotpotqa_adapter import (
        compute_exact_match, compute_f1)

    assert compute_exact_match("Paris", "Paris")
    assert not compute_exact_match("Paris", "London")
    assert compute_f1("Paris is nice", "Paris") > 0
    assert compute_f1("", "") == 0.0


def test_drop_adapter():
    config = _make_config("DROP")
    adapter = DROPAdapter(config)
    adapter.load()
    tasks = adapter.list_tasks()
    assert len(tasks) == 5
    for task_id in tasks:
        task = adapter.get_task(task_id)
        answers = task.get("answers", [])
        expected = answers[0] if answers else "0"
        agent = _FixedMockAgent(expected)
        exec_record = adapter.run(agent, task)
        assert exec_record.status == "success"
        eval_record = adapter.evaluate(exec_record)
        assert eval_record.success, f"Failed on {task_id} with answer {expected}"
        assert eval_record.score >= 0.5


def test_drop_fallback_task_structure():
    config = _make_config("DROP")
    adapter = DROPAdapter(config)
    adapter.load()
    task = adapter.get_task("drop_0")
    assert task["task_id"] == "drop_0"
    assert "passage" in task
    assert "question" in task
    assert "answers" in task
    assert isinstance(task["answers"], list)


def test_drop_normalize():
    from llm_reliability.benchmarks.adapters.drop_adapter import (
        _f1_score, _normalize_text)

    assert _normalize_text("Hello, World!") == "hello world"
    assert _normalize_text("") == ""
    assert _f1_score("hello world", "hello") > 0
    assert _f1_score("", "") == 1.0


def test_hotpotqa_normalize():
    from llm_reliability.benchmarks.adapters.hotpotqa_adapter import \
        normalize_text

    assert normalize_text("Hello, World!") == "hello world"
    assert normalize_text("") == ""
