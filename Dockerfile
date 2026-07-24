FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies for common providers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install base package
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e ".[dev]"

# Default entry point
ENTRYPOINT ["python", "-m", "llm_reliability"]
CMD ["--help"]
