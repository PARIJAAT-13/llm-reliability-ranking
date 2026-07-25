# Plugin Development Guide — LLM Reliability Ranking Framework

This guide explains how to extend the framework with custom runtimes, benchmarks, metrics, reporting, and visualizations.

---

## 1. Plugin Architecture Overview

The framework uses a registry-based plugin architecture with three core registries:

| Registry | Purpose | Base Class |
|----------|---------|------------|
| `RuntimeRegistry` | Inference backends | `Runtime` |
| `BenchmarkRegistry` | Benchmark adapters | `BenchmarkPlugin` / `BaseBenchmarkAdapter` |
| `ProviderRegistry` | Internal provider adapters | `BaseLLMAdapter` |

Each registry supports:
- **Direct registration**: `Registry.register("name", MyClass)`
- **Decorator registration**: `@Registry.register("name")`
- **Auto-discovery**: Scans well-known packages for subclasses

---

## 2. Creating a Custom Runtime

```python
from llm_reliability.runtime import Runtime
from llm_reliability.runtime.metadata import RuntimeMetadata, RuntimeCapabilities
from llm_reliability.runtime.registry import RuntimeRegistry

class MyCustomRuntime(Runtime):
    """Custom inference runtime."""

    def __init__(self, model: str = "default", **kwargs):
        self._model = model
        self._client = None

    def initialize(self) -> None:
        # Set up connections, load libraries
        self._client = ...

    def reset(self) -> None:
        pass

    def run(self, task: dict) -> str:
        prompt = task.get("prompt", "")
        return self._client.generate(prompt)

    def shutdown(self) -> None:
        self._client = None

    def metadata(self) -> dict:
        return {"runtime": "my_custom", "model": self._model}

    # Optional capabilities:
    def health_check(self) -> bool:
        return self._client is not None

    def count_tokens(self, text: str) -> int:
        return len(text.split())  # approximate

    def runtime_metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(
            runtime_name="my_custom",
            runtime_version="1.0.0",
            backend="custom",
            capabilities=RuntimeCapabilities(health_check=True, token_counting=True),
        )

# Register
RuntimeRegistry.register("my_custom", MyCustomRuntime)
```

### Runtime Capability Methods

| Method | Signature | Default | Purpose |
|--------|-----------|---------|---------|
| `load_model()` | `() -> None` | No-op | Pre-load model into memory |
| `unload_model()` | `() -> None` | No-op | Release model from memory |
| `health_check()` | `() -> bool` | Returns `True` | Verify runtime is responsive |
| `count_tokens(text)` | `(str) -> int` | Returns `0` | Tokenize without inference |
| `measure_latency(task)` | `(dict) -> tuple[Any, float]` | Calls `execute()` | Time a single inference |
| `measure_memory()` | `() -> dict[str, float]` | Returns `{}` | Report memory usage in MB |
| `runtime_metadata()` | `() -> RuntimeMetadata` | Minimal metadata | Full runtime info |

---

## 3. Creating a Custom Benchmark

```python
from llm_reliability.benchmarks.plugin import BenchmarkPlugin
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry
from llm_reliability.interfaces.agent import Agent
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.configs.config import Configuration

class MyBenchmark(BenchmarkPlugin):
    benchmark_name = "my_benchmark"

    def __init__(self, config: Configuration):
        self._tasks = {}

    def _load_tasks(self) -> None:
        self._tasks = {"task_1": {"prompt": "Solve X", "expected": "42"}}

    def list_tasks(self) -> list[str]:
        return sorted(self._tasks.keys())

    def get_task(self, task_id: str) -> dict:
        return dict(self._tasks[task_id])

    def run(self, agent: Agent, task: dict) -> ExecutionRecord:
        output = agent.run(task)
        return ExecutionRecord(
            configuration_hash="...",
            seed=42, benchmark="my_benchmark", agent="test",
            task_id=task.get("task_id", "unknown"),
            run_index=0, runtime_seconds=1.0,
            timestamp="...", stdout=str(output), stderr="",
            status="success",
        )

    def evaluate(self, execution: ExecutionRecord) -> EvaluationRecord:
        return EvaluationRecord.from_execution(
            execution=execution, success=True, score=1.0,
            evaluated_at="...",
        )

    def collect_logs(self) -> dict:
        return {}

    def metadata(self) -> dict:
        return {"name": "my_benchmark"}

BenchmarkRegistry.register("my_benchmark", MyBenchmark)
```

---

## 4. Plugin Auto-Discovery

The framework automatically discovers plugins by scanning Python packages at import time.

- **Runtime plugins**: Place subclasses of `Runtime` in `llm_reliability.agents` or any package discovered by `RuntimeRegistry.discover()`.
- **Benchmark plugins**: Place subclasses of `BaseBenchmarkAdapter` in `llm_reliability.benchmarks.adapters`.

To manually trigger discovery:

```python
RuntimeRegistry.discover()       # scans llm_reliability.agents
BenchmarkRegistry.discover()     # scans llm_reliability.benchmarks.adapters
```

---

## 5. Creating Custom Report Formats

Extend the reporting system by adding a new writer:

```python
from llm_reliability.reporting.summary import ExperimentSummary

def generate_custom_report(summary: ExperimentSummary, output_path: str) -> None:
    with open(output_path, "w") as f:
        f.write(f"# Custom Report: {summary.experiment_name}\n")
        for ranking in summary.reliability_rankings:
            f.write(f"- {ranking.agent}: {ranking.score:.4f}\n")
```

Register with the `ReportGenerator`:

```python
from llm_reliability.reporting.report_generator import ReportGenerator
ReportGenerator.register_writer("custom", generate_custom_report)
```

---

## 6. Creating Custom Visualizations

```python
import matplotlib.pyplot as plt
from llm_reliability.records.metric import MetricRecord

def plot_custom_comparison(metrics: list[MetricRecord], output_path: str) -> None:
    agents = [m.agent for m in metrics]
    scores = [m.composite_reliability for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(agents, scores)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Composite Reliability")
    ax.set_title("Custom Reliability Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

---

## 7. Configuration Schema

When extending the configuration system, use Pydantic v2 models:

```python
from pydantic import Field, field_validator
from llm_reliability.utils.serialization import SerializableModel

class MyPluginConfig(SerializableModel):
    enabled: bool = True
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    parameters: dict[str, str] = Field(default_factory=dict)

    @field_validator("threshold")
    @classmethod
    def check_threshold(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return v
```

---

## 8. Testing Guidelines

```python
import pytest
from my_plugin import MyCustomRuntime

def test_runtime_initialization():
    runtime = MyCustomRuntime(model="test")
    assert runtime._model == "test"

def test_runtime_metadata():
    runtime = MyCustomRuntime()
    meta = runtime.runtime_metadata()
    assert meta.runtime_name == "my_custom"

def test_health_check_no_server():
    runtime = MyCustomRuntime()
    assert runtime.health_check() is False
```
