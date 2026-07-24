"""
Runtime — abstract interface for interchangeable inference backends.

``Runtime`` extends ``Agent`` with an explicit ``execute()`` entry-point
so that benchmarks and the execution engine depend only on this
abstraction, not on any specific provider.

Existing agent classes that inherit from ``Agent`` can migrate by simply
changing their base class to ``Runtime`` — the default ``execute()``
implementation delegates to ``run()``, preserving full backward
compatibility.
"""

from __future__ import annotations

from abc import ABC
from typing import Any

from llm_reliability.interfaces.agent import Agent


class Runtime(Agent, ABC):
    """Abstract interface for all inference runtimes.

    Core lifecycle
    --------------
    ``initialize()`` → ``execute()`` → ``shutdown()``

    Subclasses may override ``execute()`` directly or keep inheriting
    the default which calls ``run()``.
    """

    def execute(self, task: dict[str, Any]) -> Any:
        """Execute a task and return the raw output.

        The default implementation delegates to ``run()`` so that
        existing agents (which implement ``run()``) continue to work
        without modification.
        """
        return self.run(task)
