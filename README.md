# LLM Reliability Ranking Framework

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/PARIJAAT-13/llm-reliability-ranking/actions/workflows/ci.yml/badge.svg)](https://github.com/PARIJAAT-13/llm-reliability-ranking/actions/workflows/ci.yml)

An extensible research framework for benchmarking, evaluating, and ranking local Large Language Models (LLMs) across multi-run execution matrices, perturbation suites, and system fault scenarios under physical hardware constraints. This repository contains the complete experimental pipeline and analysis code for the paper:

> **LLM Reliability Ranking Framework: Benchmarking Local Open-Weights Language Models Under Hardware Memory Constraints**

---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing Published Results](#reproducing-published-results)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

---

## Key Features

- **Memory-Aware Execution Engine** — Detects non-retryable memory allocation failures (`OllamaMemoryError`), fast-skips memory-constrained models, and unloads VRAM/RAM model weights via `keep_alive: 0`.
- **Decoupled System Prompt Architecture** — Injects task formatting instructions at the experiment configuration layer without polluting benchmark adapter code.
- **Heterogeneous Benchmark Adapters** — Built-in support for GAIA, AgentBoard, SWE-Bench Lite, ARC, GSM8K, MMLU, Hellaswag, PIQA, WinoGrande, TruthfulQA, MBPP, HumanEval, and custom mock suites.
- **Multi-Dimensional Reliability Metrics** — Evaluates Success Rate ($S$), Repeated-Run Consistency ($C$), Perturbation Robustness ($R$), and Fault Tolerance ($F$).
- **Plugin-Based Architecture** — Register custom benchmarks and agent runtimes via `BenchmarkRegistry` and `RuntimeRegistry` without editing core files.
- **Statistical Validation** — Bootstrap confidence intervals, Spearman/Kendall correlations, hypothesis testing (independent t-test), ablation analysis, and sensitivity analysis.
- **Canonical Serialization** — Pydantic v2 schemas with incremental checkpointing for resumable experiments.
- **Multi-Format Reporting** — Export results as Markdown, LaTeX, and HTML with publication-quality figures (PNG, SVG, PDF).

---

## Installation

### Prerequisites

- Python 3.10+
- Local [Ollama](https://ollama.com/) instance (`http://127.0.0.1:11434`) for local model inference
- (Optional) API keys for OpenAI, Anthropic, Gemini, DeepSeek, or other cloud providers

### Setup

```bash
git clone https://github.com/PARIJAAT-13/llm-reliability-ranking.git
cd llm-reliability-ranking
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

Install optional dependency groups as needed:

```bash
pip install -e ".[dev]"         # Testing (pytest)
pip install -e ".[visualization]" # Plotting (matplotlib, seaborn, pandas)
pip install -e ".[openai]"      # OpenAI provider
pip install -e ".[anthropic]"   # Anthropic provider
pip install -e ".[providers]"   # All cloud providers
```

---

## Supported Models

The framework includes a built-in model registry covering **54 open-weight language models** across **14 model families**, all compatible with Ollama's local inference API.

### Model Families

| Family | Supported Sizes | Example Pull Command |
|--------|----------------|---------------------|
| **Llama** (Meta) | 1B, 3B, 7B, 8B, 13B, 34B, 70B | `ollama pull llama3.1:8b` |
| **Qwen** (Alibaba) | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B | `ollama pull qwen2.5:7b` |
| **Gemma** (Google) | 2B, 7B, 9B, 27B | `ollama pull gemma2:9b` |
| **Phi** (Microsoft) | 3.8B, 14B | `ollama pull phi4:14b` |
| **Mistral** (Mistral AI) | 7B, 12B, 46.7B (MoE) | `ollama pull mistral:7b` |
| **DeepSeek** | 7B, 14B, 16B, 32B, 70B | `ollama pull deepseek-r1:7b` |
| **StarCoder2** (ServiceNow) | 3B, 7B, 15B | `ollama pull starcoder2:7b` |
| **Yi** (01.AI) | 1.5B, 6B, 9B | `ollama pull yi:9b` |
| **Falcon** (TII) | 7B | `ollama pull falcon3:7b` |
| **+ more** | TinyLlama, SmolLM, OpenChat, DBRX, etc. | |

See [paper/model_table.md](paper/model_table.md) for the complete publication-ready catalogue.

### Using the Model Registry

The `ModelRegistry` provides programmatic access to all supported models:

```python
from llm_reliability.models import populate_registry, ModelRegistry, ModelInfo

# Populate the registry with all supported models
populate_registry()

# Look up a model by Ollama identifier
model = ModelRegistry.get("llama3.1:8b")
print(f"Family: {model.family}, Parameters: {model.parameters}, Context: {model.context_window}")

# List all models
for model_id in ModelRegistry.list_identifiers():
    info = ModelRegistry.get(model_id)
    print(f"{model_id:40s} {info.family:15s} {info.parameters}")

# List by family
by_family = ModelRegistry.list_by_family()
for family, models in by_family.items():
    print(f"{family}: {len(models)} models")
```

### Adding New Models

To add a new Ollama model, create a `ModelInfo` instance and register it:

```python
from llm_reliability.models import ModelInfo, ModelRegistry

my_model = ModelInfo(
    family="MyFamily",
    name="My Model 7B",
    parameters="7B",
    parameter_count=7.0,
    context_window=4096,
    recommended_ram_gb=8.0,
    recommended_vram_gb=6.0,
    ollama_identifier="my-model:7b",
)
ModelRegistry.register(my_model)
```

To permanently add a model to the built-in catalogue, add its `ModelInfo` to the
`SUPPORTED_OLLAMA_MODELS` list in `src/llm_reliability/models/ollama_models.py`.

### Validation

The registry validates all models at registration time:

- **Duplicate detection** — same `ollama_identifier` cannot be registered twice
- **Identifier format** — must contain a colon (`family:size`)
- **Type safety** — only `ModelInfo` instances accepted
- **Parameter validation** — `parameter_count` must be non-negative

### Automatic Discovery (Optional)

The `discover_local_models()` function queries a running Ollama instance for
locally installed models and merges them into the registry:

```python
from llm_reliability.models import discover_local_models, merge_discovered

discovered = discover_local_models()
count = merge_discovered(discovered)
print(f"Discovered {count} new models from local Ollama")
```

This feature is optional and does not replace static configuration.
It never modifies or removes existing registry entries.

---

## Quick Start

Run a self-contained experiment with a mock benchmark and mock agent:

```bash
python examples/basic_experiment.py
```

Or via the CLI:

```bash
# List available benchmarks
llm-reliability list benchmarks

# Validate an experiment configuration
llm-reliability validate configs/demo_full_experiment_spec.json

# Run a configured experiment
llm-reliability run configs/demo_full_experiment_spec.json
```

---

## Reproducing Published Results

To reproduce the full-scale evaluation study on local Ollama models (`llama3.1:8b`, `mistral:7b`, `qwen2.5:7b`, `gemma2:9b`) against the GAIA dataset:

```bash
# 1. Pull models
ollama pull qwen2.5:7b
ollama pull gemma2:9b
ollama pull mistral:7b
ollama pull llama3.1:8b

# 2. Run large-scale experiment
python scripts/run_large_scale_experiment.py \
  --config configs/full_experiment_config.json \
  --output-dir results/full_study

# 3. Generate publication artifacts
python scripts/generate_report.py --input-dir results/full_study
```

Output artifacts under `results/full_study/<experiment_id>/`:

| File | Description |
|------|-------------|
| `configuration.json` | Immutable experiment specification with SHA-256 hash |
| `executions.json` | Per-task runtime logs, latency, token counts |
| `evaluations.json` | Per-task correctness scores |
| `metrics.json` | Aggregated success rate, consistency, reliability scores |
| `rankings.json` | Final ordinal ranking tables |
| `checkpoint.json` | Execution recovery state |
| `manifest.json` | Artifact SHA-256 hashes for verification |

---

## Project Structure

```
├── src/llm_reliability/     # Framework source code
│   ├── agents/              # LLM agent adapters (Ollama, OpenAI, etc.)
│   ├── benchmarks/          # Benchmark adapters (GAIA, AgentBoard, etc.)
│   ├── configs/             # Experiment configuration models
│   ├── experiments/         # Experiment manager and runner
│   ├── metrics/             # Reliability metric computation
│   ├── models/              # Model registry and metadata catalogue
│   ├── pipeline/            # Core orchestration pipeline
│   ├── ranking/             # Success and reliability ranking engines
│   ├── statistics/          # Statistical validation toolkit
│   └── reproducibility/     # Manifest, archive, environment capture
├── tests/                   # Pytest test suite (920+ tests)
├── paper/                   # Research publication artifacts
│   ├── sections/            # LaTeX manuscript sections
│   ├── figures/             # Publication figures (PDF, PNG)
│   └── tables/              # LaTeX result tables
├── scripts/                 # Experiment runners and report generators
├── configs/                 # Experiment configuration templates
├── docs/                    # Documentation
├── examples/                # Runnable examples
└── legacy/                  # Deprecated scripts (archived)
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Component architecture and data flow |
| [Developer Guide](docs/developer_guide.md) | Extending benchmarks, agents, and pipelines |
| [Reproducibility](docs/reproducibility.md) | Experiment reproduction and verification |
| [Deployment](docs/DEPLOYMENT.md) | Multi-platform deployment (local, cloud, edge) |

### Paper Artifacts

| Document | Description |
|----------|-------------|
| [Manuscript](paper/manuscript.md) | Complete research paper |
| [Methodology](paper/methodology.md) | System architecture and metric formulations |
| [Results](paper/results.md) | Quantitative tables and evaluation matrices |
| [Audit Report](paper/audit_report.md) | Technical audit of data integrity |

---

## Citation

If you use this framework or its empirical results in your research:

```bibtex
@software{Parijaat_LLM_Reliability_Ranking_2026,
  author = {Parijaat Srivastava and Pooja Mourya},
  title = {LLM Reliability Ranking Framework: Benchmarking Local Open-Weights Language Models Under Hardware Memory Constraints},
  url = {https://github.com/PARIJAAT-13/llm-reliability-ranking},
  version = {1.0.0},
  year = {2026}
}
```

A `CITATION.cff` file is also included in the repository root.

---

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

## Authors

- **Parijaat Srivastava**
- **Pooja Mourya**

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, and note our [Code of Conduct](CODE_OF_CONDUCT.md). For security issues, refer to [SECURITY.md](SECURITY.md).
