"""Tests for Agent interface (Artifact 3)."""

import inspect

import pytest

from llm_reliability.interfaces import Agent


def test_agent_is_abstract() -> None:
    with pytest.raises(TypeError):
        Agent()


def test_agent_defines_required_methods() -> None:
    required = {"initialize", "reset", "run", "shutdown", "metadata"}
    method_names = {
        name for name, member in inspect.getmembers(Agent, predicate=inspect.isfunction)
    }
    assert required.issubset(method_names)
    for name in required:
        assert getattr(Agent, name).__isabstractmethod__


class _MinimalAgent(Agent):
    def initialize(self) -> None:
        return None

    def reset(self) -> None:
        return None

    def run(self, task: dict[str, str]) -> str:
        return task["id"]

    def shutdown(self) -> None:
        return None

    def metadata(self) -> dict[str, str]:
        return {"name": "minimal"}


def test_concrete_agent_can_be_instantiated() -> None:
    agent = _MinimalAgent()
    assert agent.run({"id": "task-1"}) == "task-1"
