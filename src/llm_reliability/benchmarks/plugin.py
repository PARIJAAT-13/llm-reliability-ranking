"""
BenchmarkPlugin — abstract base class for self-registering benchmark adapters.

Adapters that extend ``BenchmarkPlugin`` (or its subclass
``BaseBenchmarkAdapter``) can be auto-discovered and registered without
modifying any core framework file.  Each plugin sets ``benchmark_name``
to the canonical name used for lookup in ``BenchmarkRegistry``.

Usage::

    from llm_reliability.benchmarks import BenchmarkPlugin

    class MyAdapter(BenchmarkPlugin):
        benchmark_name = "MyBenchmark"
        ...
"""

from abc import ABC
from typing import ClassVar

from llm_reliability.interfaces.benchmark import Benchmark


class BenchmarkPlugin(Benchmark, ABC):
    """Abstract base for self-registering benchmark adapters.

    Subclasses must set ``benchmark_name`` to the canonical name that
    will be used when looking up the adapter in ``BenchmarkRegistry``.
    """

    benchmark_name: ClassVar[str] = ""
