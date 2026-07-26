"""
Purpose
-------
Provides reusable infrastructure and registry for integrating real-world benchmarks.

Adapters can be added as new modules in this package and will be
automatically discovered — no core file edits required.
"""

from __future__ import annotations

from llm_reliability.benchmarks.adapters.base_adapter import BaseBenchmarkAdapter
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry

# Auto-discover all benchmark adapter modules in this package.
# Each module's module-level registration call registers the adapter.
BenchmarkRegistry.discover()

# Re-export for backward compatibility with existing imports.
# Third-party code should prefer ``BenchmarkRegistry.get("name")``.
from llm_reliability.benchmarks.adapters.agentboard_adapter import AgentBoardAdapter
from llm_reliability.benchmarks.adapters.arc_adapter import ARCAdapter
from llm_reliability.benchmarks.adapters.arc_challenge_adapter import (
    ARCChallengeAdapter,
)
from llm_reliability.benchmarks.adapters.arena_hard_adapter import ArenaHardAdapter
from llm_reliability.benchmarks.adapters.bbh_adapter import BBHAdapter
from llm_reliability.benchmarks.adapters.bigbench_lite_adapter import (
    BigBenchLiteAdapter,
)
from llm_reliability.benchmarks.adapters.boolq_adapter import BoolQAdapter
from llm_reliability.benchmarks.adapters.commonsenseqa_adapter import (
    CommonsenseQAAdapter,
)
from llm_reliability.benchmarks.adapters.drop_adapter import DROPAdapter
from llm_reliability.benchmarks.adapters.gaia_adapter import GAIAAdapter
from llm_reliability.benchmarks.adapters.gpqa_adapter import GPQAAdapter
from llm_reliability.benchmarks.adapters.gsm8k_adapter import GSM8KAdapter
from llm_reliability.benchmarks.adapters.hellaswag_adapter import HellaSwagAdapter
from llm_reliability.benchmarks.adapters.hotpotqa_adapter import HotpotQAAdapter
from llm_reliability.benchmarks.adapters.humaneval_adapter import HumanEvalAdapter
from llm_reliability.benchmarks.adapters.ifeval_adapter import IFEvalAdapter
from llm_reliability.benchmarks.adapters.livecodebench_adapter import (
    LiveCodeBenchAdapter,
)
from llm_reliability.benchmarks.adapters.mbpp_adapter import MBPPAdapter
from llm_reliability.benchmarks.adapters.mmlu_adapter import MMLUAdapter
from llm_reliability.benchmarks.adapters.mmlu_pro_adapter import MMLUProAdapter
from llm_reliability.benchmarks.adapters.natural_questions_adapter import (
    NaturalQuestionsAdapter,
)
from llm_reliability.benchmarks.adapters.openbookqa_adapter import OpenBookQAAdapter
from llm_reliability.benchmarks.adapters.piqa_adapter import PIQAAdapter
from llm_reliability.benchmarks.adapters.reliability_bench_adapter import (
    ReliabilityBenchAdapter,
)
from llm_reliability.benchmarks.adapters.swebench_lite_adapter import (
    SWEBenchLiteAdapter,
)
from llm_reliability.benchmarks.adapters.triviaqa_adapter import TriviaQAAdapter
from llm_reliability.benchmarks.adapters.truthfulqa_adapter import TruthfulQAAdapter
from llm_reliability.benchmarks.adapters.webarena_adapter import WebArenaAdapter
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
    "ReliabilityBenchAdapter",
    "MMLUProAdapter",
    "ARCChallengeAdapter",
    "BBHAdapter",
    "DROPAdapter",
    "BoolQAdapter",
    "CommonsenseQAAdapter",
    "OpenBookQAAdapter",
    "GPQAAdapter",
    "TriviaQAAdapter",
    "NaturalQuestionsAdapter",
    "HotpotQAAdapter",
    "IFEvalAdapter",
    "BigBenchLiteAdapter",
    "ArenaHardAdapter",
    "LiveCodeBenchAdapter",
    "WebArenaAdapter",
]
