"""Runtime abstraction layer — interchangeable inference backends."""

from __future__ import annotations

from llm_reliability.runtime.batching import (
    AdaptiveBatcher,
    BatchExecutor,
    BatchProcessor,
    BatchResult,
    BatchStatistics,
)
from llm_reliability.runtime.cost_accounting import (
    CostCalculator,
    CostEntry,
    CostTracker,
    TokenAccount,
    TokenUsage,
)
from llm_reliability.runtime.failover import (
    FailoverConfig,
    FailoverResult,
    FailoverStrategy,
    ProviderFailover,
    RetryConfig,
    RetryExecutor,
    RetryResult,
    RetryStrategy,
)
from llm_reliability.runtime.hardware_profiler import (
    CPUInfo,
    GPUInfo,
    HardwareProfile,
    HardwareProfiler,
    MemoryInfo,
    RuntimeStatistics,
    RuntimeTimer,
)
from llm_reliability.runtime.interface import Runtime
from llm_reliability.runtime.metadata import RuntimeCapabilities, RuntimeMetadata
from llm_reliability.runtime.provider_base import BaseProvider
from llm_reliability.runtime.registry import RuntimeRegistry
from llm_reliability.runtime.streaming import (
    StreamAdapter,
    StreamingExecutor,
    StreamStatistics,
    TokenCollector,
    TokenStream,
    TokenStreamCollector,
)

__all__ = [
    "Runtime",
    "RuntimeRegistry",
    "RuntimeCapabilities",
    "RuntimeMetadata",
    "BaseProvider",
    "BatchProcessor",
    "BatchExecutor",
    "BatchResult",
    "BatchStatistics",
    "AdaptiveBatcher",
    "StreamingExecutor",
    "StreamAdapter",
    "TokenStream",
    "TokenCollector",
    "TokenStreamCollector",
    "StreamStatistics",
    "RetryExecutor",
    "RetryConfig",
    "RetryResult",
    "RetryStrategy",
    "ProviderFailover",
    "FailoverConfig",
    "FailoverResult",
    "FailoverStrategy",
    "CostCalculator",
    "CostEntry",
    "CostTracker",
    "TokenAccount",
    "TokenUsage",
    "HardwareProfiler",
    "HardwareProfile",
    "CPUInfo",
    "GPUInfo",
    "MemoryInfo",
    "RuntimeStatistics",
    "RuntimeTimer",
]
