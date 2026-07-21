"""
Agent Contract Tests

Tests whether an Agent implementation fulfills its contract.
Includes both positive tests against a valid agent and negative tests
against intentionally broken agent implementations.
"""

import copy

import pytest

from llm_reliability.interfaces.agent import Agent


class ValidAgent(Agent):
    """A valid agent implementation that fulfills the contract."""
    def initialize(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def run(self, task: dict) -> str:
        if not task:
            raise ValueError("Invalid task")
        return "Deterministic Answer"

    def shutdown(self) -> None:
        pass

    def metadata(self) -> dict:
        return {"name": "valid"}


class BrokenAgentNoInterface(Agent):
    """Fails to implement required abstract methods."""
    pass


class BrokenAgentMutatesTask(Agent):
    """Intentionally breaks the contract by mutating the input task."""
    def initialize(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def run(self, task: dict) -> str:
        task["mutated"] = True
        return "Answer"

    def shutdown(self) -> None:
        pass

    def metadata(self) -> dict:
        return {"name": "broken"}


def test_agent_implements_interface():
    agent = ValidAgent()
    assert isinstance(agent, Agent)

    with pytest.raises(TypeError):
        BrokenAgentNoInterface()


def test_agent_deterministic():
    agent1 = ValidAgent()
    ans1 = agent1.run({"task_id": "1"})
    
    agent2 = ValidAgent()
    ans2 = agent2.run({"task_id": "1"})

    assert ans1 == ans2


def test_agent_valid_outputs():
    agent = ValidAgent()
    ans = agent.run({"task_id": "1"})
    assert ans is not None

    meta = agent.metadata()
    assert isinstance(meta, dict)


def test_agent_invalid_input():
    agent = ValidAgent()
    with pytest.raises(Exception):
        agent.run({})


def test_broken_agent_mutates_task():
    agent = BrokenAgentMutatesTask()
    task = {"task_id": "1"}
    original_task = copy.deepcopy(task)
    
    # Contract: Never mutate input tasks. 
    # This broken agent violates it.
    agent.run(task)
    assert task != original_task
    assert "mutated" in task
