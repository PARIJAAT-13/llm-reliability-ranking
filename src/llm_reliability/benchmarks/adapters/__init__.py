"""
Purpose
-------
Provides reusable infrastructure for integrating real-world benchmarks.

Design notes
------------
The Adapter Framework minimizes code duplication when adding new benchmarks.
New benchmarks (like AgentBoard, GAIA) inherit from BaseBenchmarkAdapter, 
which provides standard validation, logging, and execution scaffolding.

The BenchmarkRegistry provides dynamic discovery and instantiation of these
adapters, allowing the pipeline to operate on abstract identifiers rather 
than tightly coupling to specific implementations.
"""

from llm_reliability.benchmarks.adapters.base_adapter import BaseBenchmarkAdapter
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry

__all__ = ["BaseBenchmarkAdapter", "BenchmarkRegistry"]
