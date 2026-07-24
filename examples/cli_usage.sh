#!/usr/bin/env bash
# Example CLI commands for the llm-reliability framework.
# Run these from the repository root after `pip install -e .`

# Install the framework in editable mode
pip install -e .

# Run an experiment from a JSON config file
llm-reliability run config.json

# List all registered benchmark adapters
llm-reliability list benchmarks

# List all registered runtimes (agent types)
llm-reliability list runtimes

# Validate an experiment configuration file
llm-reliability validate config.json

# Clear the experiment result cache
llm-reliability clear-cache

# Show version
llm-reliability --version

# Launch with Docker (requires Docker and docker-compose)
docker compose up
