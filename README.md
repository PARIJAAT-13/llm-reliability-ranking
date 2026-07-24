# LLM Reliability Ranking Framework

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-347%20passed-brightgreen.svg)](https://docs.pytest.org/)

An extensible, production-ready research framework for benchmarking, evaluating, and ranking local Large Language Models (LLMs) across multi-run execution matrices, perturbation suites, and system fault scenarios under physical hardware constraints.

---

## 🌟 Key Features

- **Memory-Aware Execution Engine**: Detects non-retryable memory allocation failures (`OllamaMemoryError`), fast-skips memory-constrained models in 0.0 seconds, and automatically unloads VRAM/RAM model weights via `keep_alive: 0`.
- **Decoupled System Prompt Architecture**: Injects task formatting instructions (`system_prompt`) at the experiment configuration layer without polluting benchmark adapter code.
- **Heterogeneous Benchmark Adapters**: Built-in support for **GAIA**, **AgentBoard**, **SWE-Bench Lite**, and custom mock benchmark suites.
- **Multi-Dimensional Reliability Metrics**: Evaluates **Success Rate ($S$)**, **Repeated-Run Consistency ($C$)**, **Perturbation Robustness ($R$)**, and **Fault Tolerance ($F$)**.
- **Canonical Serialization & Resumable Checkpoints**: Serializes executions, evaluations, metrics, and rankings via Pydantic v2 schemas with incremental checkpointing (`checkpoint.json`).

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Local [Ollama](https://ollama.com/) instance (`http://127.0.0.1:11434`)

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/parijaat/llm-reliability-ranking.git
cd llm-reliability-ranking

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies in editable mode
pip install -e .
```

---

## 🔬 Reproducing Published Experiment Results

To execute the large-scale evaluation study on local Ollama models (`llama3.1:8b`, `mistral:7b`, `qwen2.5:7b`, `gemma2:9b`) against the GAIA dataset:

### Step 1: Ensure Candidate Models are Available in Ollama

```bash
ollama pull qwen2.5:7b
ollama pull gemma2:9b
ollama pull mistral:7b
ollama pull llama3.1:8b
```

### Step 2: Run Large-Scale Experiment Script

```bash
python scripts/run_large_scale_experiment.py \
  --config configs/full_experiment_config.json \
  --output-dir results/full_study
```

### Step 3: Inspect Output Artifacts

Execution outputs will be saved under `results/full_study/<experiment_id>/`:
- `configuration.json`: Immutable experiment specification and configuration SHA-256 hash.
- `executions.json`: Per-task runtime logs, latency (`runtime_seconds`), prompt token counts, and outputs.
- `evaluations.json`: Per-task exact-match correctness scores.
- `metrics.json`: Aggregated success rate, consistency, and composite reliability scores per model.
- `rankings.json`: Final ordinal ranking tables.
- `checkpoint.json`: Execution recovery state.

---

## 🧪 Running the Verification & Test Suite

To verify codebase integrity, run the automated linter and pytest suite:

```bash
# Code style and linting check
ruff check src tests

# Run unit and integration tests (347 test cases)
pytest
```

---

## 📊 Published Research Artifacts

The repository includes complete publication artifacts under `paper/`:
- **`paper/manuscript.md`**: Complete camera-ready research paper.
- **`paper/results.md`**: Granular quantitative tables and per-task evaluation matrices.
- **`paper/methodology.md`**: System architecture, memory fast-failing, and metric formulations.
- **`paper/audit_report.md`**: Complete technical audit report validating data integrity.
- **`paper/figures/`**: 300 DPI publication plots (`accuracy.png`, `reliability.png`, `latency.png`, `completion_rate.png`, `ranking.png`).

---

## 📜 Citation

If you use this framework or empirical benchmark dataset in your research, please cite:

```bibtex
@software{Parijaat_LLM_Reliability_Ranking_2026,
  author = {Parijaat},
  title = {LLM Reliability Ranking Framework: Evaluating Local Language Models Under Memory Constraints},
  url = {https://github.com/parijaat/llm-reliability-ranking},
  version = {1.0.0},
  year = {2026}
}
```

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

## Authors

- Parijaat Srivastava
- Pooja Mourya

---

Developed as part of the research project

**LLM Reliability Ranking Framework: Benchmarking Local Open-Weights Language Models Under Hardware Memory Constraints**
