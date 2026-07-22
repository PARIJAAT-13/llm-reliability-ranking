# LLM Reliability Ranking Framework

Research framework for empirically comparing success-based and reliability-based
rankings of LLM agents across multiple benchmarks.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

## Project structure

```
src/llm_reliability/
    configs/       # Immutable experiment configuration
    interfaces/    # Benchmark and Agent contracts
    records/       # Execution → Evaluation → Metric → Ranking pipeline records
    utils/         # Shared serialization utilities
tests/             # Unit, negative, and round-trip tests
```

## Implementation status

| Artifact | Module | Status |
|----------|--------|--------|
| 1 | Configuration | Complete |
| 2 | Benchmark Interface | Complete |
| 3 | Agent Interface | Complete |
| 4 | ExecutionRecord | Complete |
| 5 | EvaluationRecord | Complete |
| 6 | MetricRecord | Complete |
| 7 | RankingRecord | Complete |
