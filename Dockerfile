FROM python:3.14-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir build && python -m build --wheel
RUN pip install --no-cache-dir dist/*.whl --target /install

FROM python:3.14-slim

WORKDIR /app

RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

COPY --from=builder /install /usr/local/lib/python3.11/site-packages

RUN mkdir -p /app/results /app/configs /app/.cache && \
    chown -R appuser:appuser /app

USER appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import llm_reliability; print('OK')" || exit 1

ENTRYPOINT ["python", "-m", "llm_reliability"]
CMD ["--help"]
