"""
Purpose
-------
Provides reusable infrastructure and registry for integrating real-world benchmarks.

Adapters can be added as new modules in this package and will be
automatically discovered — no core file edits required.
"""

from llm_reliability.benchmarks.adapters.base_adapter import BaseBenchmarkAdapter
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry

# Auto-discover all benchmark adapter modules in this package.
# Each module's module-level registration call registers the adapter.
BenchmarkRegistry.discover()

# Re-export for backward compatibility with existing imports.
# Third-party code should prefer ``BenchmarkRegistry.get("name")``.
from llm_reliability.benchmarks.adapters.agentboard_adapter import AgentBoardAdapter
from llm_reliability.benchmarks.adapters.arc_adapter import ARCAdapter
from llm_reliability.benchmarks.adapters.gaia_adapter import GAIAAdapter
from llm_reliability.benchmarks.adapters.gsm8k_adapter import GSM8KAdapter
from llm_reliability.benchmarks.adapters.hellaswag_adapter import HellaSwagAdapter
from llm_reliability.benchmarks.adapters.humaneval_adapter import HumanEvalAdapter
from llm_reliability.benchmarks.adapters.mbpp_adapter import MBPPAdapter
from llm_reliability.benchmarks.adapters.mmlu_adapter import MMLUAdapter
from llm_reliability.benchmarks.adapters.piqa_adapter import PIQAAdapter
from llm_reliability.benchmarks.adapters.swebench_lite_adapter import (
    SWEBenchLiteAdapter,
)
from llm_reliability.benchmarks.adapters.truthfulqa_adapter import TruthfulQAAdapter
from llm_reliability.benchmarks.adapters.winogrande_adapter import WinograndeAdapter

__all__ = [
    "BaseBenchmarkAdapter",
    "BenchmarkRegistry",
    "AgentBoardAdapter",
    "GAIAAdapter",
    "SWEBenchLiteAdapter",
    "MMLUAdapter",
    "HellaSwagAdapter",
    "HumanEvalAdapter",
    "MBPPAdapter",
    "TruthfulQAAdapter",
    "GSM8KAdapter",
    "ARCAdapter",
    "WinograndeAdapter",
    "PIQAAdapter",
]
