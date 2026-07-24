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
│   ├── pipeline/            # Core orchestration pipeline
│   ├── ranking/             # Success and reliability ranking engines
│   ├── statistics/          # Statistical validation toolkit
│   └── reproducibility/     # Manifest, archive, environment capture
├── tests/                   # Pytest test suite (360+ tests)
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
