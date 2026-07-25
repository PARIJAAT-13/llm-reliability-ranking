# Migration Guide — LLM Reliability Ranking v1.0 → v2.0

## Overview

Version 2.0 introduces multi-runtime support, publication-ready reporting, enhanced CLI, reproducibility manifests, and plugin extensibility. All existing APIs remain backward compatible.

---

## What Changed

### New Runtime Interface

The `Runtime` base class now includes optional capability methods:

| Method | v1.0 | v2.0 |
|--------|------|------|
| `execute(task)` | ✅ | ✅ (unchanged) |
| `load_model()` | ❌ | ✅ (default: no-op) |
| `unload_model()` | ❌ | ✅ (default: no-op) |
| `health_check()` | ❌ | ✅ (default: `True`) |
| `count_tokens(text)` | ❌ | ✅ (default: `0`) |
| `measure_latency(task)` | ❌ | ✅ (calls `execute()`) |
| `measure_memory()` | ❌ | ✅ (default: `{}`) |
| `runtime_metadata()` | ❌ | ✅ (default: minimal) |

**Impact**: None. Existing agents continue to work unchanged. The methods default to safe no-ops.

### New Runtime Adapters

New standalone runtime implementations available:

| Adapter | Registration | Backward Compatible |
|---------|-------------|-------------------|
| `OllamaRuntime` | `runtime/adapters/ollama.py` | Yes (alongside existing `OllamaAgent`) |
| `VLLMRuntime` | `runtime/adapters/vllm.py` | Yes |
| `LlamaCppRuntime` | `runtime/adapters/llama_cpp.py` | Yes |
| `TGIRuntime` | `runtime/adapters/tgi.py` | Yes |
| `LMStudioRuntime` | `runtime/adapters/lm_studio.py` | Yes |
| `MLXRuntime` | `runtime/adapters/mlx.py` | Yes |
| `OpenAICompatRuntime` | `runtime/adapters/openai_compat.py` | Yes |

### New CLI Commands

| Command | Description |
|---------|-------------|
| `resume <dir>` | Resume a checkpointed experiment |
| `checkpoint <dir>` | Show checkpoint status |
| `compare <dir1> <dir2> ...` | Compare experiments |
| `report <dir>` | Generate publication artifacts |
| `export <dir>` | Export rankings (CSV/LaTeX/MD) |
| `discover-models` | Runtime model discovery hints |
| `discover-runtimes` | List registered runtimes |
| `hardware-info` | Show hardware profile |
| `system-info` | Show system information |
| `statistics <dir>` | Show experiment statistics |

### New Configuration Fields

`ExperimentRunConfig` adds:

- `runtime: str` — Select inference runtime
- `hardware_profile_id: str` — Select hardware profile
- `sweep: SweepConfig` — Parameter sweeps
- `model_groups: list[ModelGroup]` — Named model groups
- `resource_limits: ResourceLimits` — Resource constraints
- `checkpoint_frequency: int` — Runs between checkpoints
- `execution_limit: int` — Max total runs

### Publication Artifacts

The `report` and `export` commands generate:

- `experiment_summary.json`
- `runtime_summary.json`
- `hardware_summary.json`
- `benchmark_summary.json`
- `ranking_summary.json`
- `statistics_summary.json`
- `rankings.tex` (LaTeX table)
- `rankings.md` (Markdown table)
- `rankings.csv`
- `reproducibility_manifest.json`

---

## Migration Steps

### 1. Update imports (if using new features)

```python
# Old
from llm_reliability.runtime import Runtime

# New (backward compatible)
from llm_reliability.runtime import Runtime, RuntimeMetadata, RuntimeCapabilities
```

### 2. For custom runtime plugins

Add `metadata()` method if missing:

```python
def metadata(self) -> dict:
    return {"runtime": "my_runtime", "model": self._model}
```

### 3. For custom CLI scripts

Add the `compare` command if you were relying on the exact shape of `parse_args()`. The new CLI preserves all existing command signatures.

### 4. Configuration files

Experiment configs remain fully compatible. The new `ExperimentRunConfig` wraps `ExperimentSpec` with runtime/hardware/sweep fields.

---

## Deprecations

- The original `Runtime` class with only `execute()` is deprecated but supported.
- The `list` command now shows benchmarks and runtimes only (previously included experiments).
- The `RUNTIME_REGISTRY` dict in `AgentFactory` is replaced by `RuntimeRegistry`.

---

## New Dependencies

- `llama.cpp` adapter: `httpx` (optional)
- `MLX` adapter: `mlx-lm` (optional)
- All other adapters use existing `openai` SDK dependency.

No new required dependencies.
