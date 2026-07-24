# Examples

This directory contains runnable examples demonstrating the LLM Reliability Ranking Framework.

## Contents

- **basic_experiment.py** — Self-contained experiment using mock benchmark and mock agent
- **cli_usage.sh** — Example CLI commands for the `llm-reliability` tool
- **docker_usage.md** — Running experiments with Docker

## Prerequisites

```bash
pip install -e ".[dev]"
```

For the CLI examples, you also need a valid experiment config JSON.

## Quick Start

```bash
python examples/basic_experiment.py
```

This runs a full experiment pipeline with a deterministic mock benchmark (10 tasks) and mock agent, then prints the results, metrics, and rankings.

## CLI Examples

```bash
# List available benchmarks
llm-reliability list benchmarks

# Validate a config file
llm-reliability validate configs/example_config.json

# Run an experiment
llm-reliability run configs/example_config.json

# Clear cache
llm-reliability clear-cache
```
