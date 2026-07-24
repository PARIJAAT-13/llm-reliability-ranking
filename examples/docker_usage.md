# Docker Usage

The framework includes Docker support for reproducible, containerized experiment execution.

## Prerequisites

- Docker Engine 24+
- Docker Compose v2+

## Quick Start

```bash
# Build and start the container
docker compose up

# Run an experiment inside the container
docker compose exec llm-reliability llm-reliability run config.json

# List benchmarks inside the container
docker compose exec llm-reliability llm-reliability list benchmarks
```

## Building the Image Manually

```bash
docker build -t llm-reliability .
docker run --rm llm-reliability llm-reliability --version
```

## Volumes

The Docker Compose setup mounts the repository root at `/app` and exposes port `11434` for Ollama connectivity. Experiment results written to `results/` persist on the host.
