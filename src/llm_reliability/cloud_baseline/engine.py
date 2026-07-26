from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from llm_reliability.agents.anthropic_agent import AnthropicAgent
from llm_reliability.agents.gemini_agent import GeminiAgent
from llm_reliability.agents.gpt_agent import GPTAgent
from llm_reliability.cloud_baseline.models import CloudBaselineResult
from llm_reliability.configs.config import Configuration

logger = logging.getLogger(__name__)

_CLOUD_AGENT_MAP: dict[str, type] = {
    "openai": GPTAgent,
    "gpt": GPTAgent,
    "anthropic": AnthropicAgent,
    "claude": AnthropicAgent,
    "google": GeminiAgent,
    "gemini": GeminiAgent,
}

CLOUD_PROVIDERS: dict[str, list[str]] = {
    "openai": ["gpt-4o", "gpt-4.1", "o3-mini"],
    "anthropic": ["claude-3-5-sonnet", "claude-4-sonnet"],
    "google": ["gemini-2.5-pro", "gemini-2.5-flash"],
}


class CloudBaselineEngine:
    def __init__(self, config: Configuration | None = None) -> None:
        if config is None:
            config = Configuration(
                experiment_name="cloud_baseline",
                benchmark="GAIA",
                agent="auto",
                llm="auto",
                prompt_version="1",
                dataset_version="1",
                seed=42,
                repetitions=1,
                metadata={"dataset_path": ""},
            )
        self._config = config
        self._results: list[CloudBaselineResult] = []

    def run_single(
        self,
        provider: str,
        model: str,
        benchmark_name: str,
        task: dict[str, Any],
    ) -> CloudBaselineResult:
        canonical = provider.lower()
        agent_cls = _CLOUD_AGENT_MAP.get(canonical)
        if agent_cls is None:
            raise ValueError(
                f"Unknown cloud provider: {provider}. "
                f"Available: {list(_CLOUD_AGENT_MAP.keys())}"
            )

        meta: dict[str, Any] = {
            "model": model,
            "temperature": getattr(self._config, "temperature", 0.0),
            "max_tokens": getattr(self._config, "max_tokens", 1024),
        }
        cfg = self._config.model_copy(update={"metadata": meta})
        agent = agent_cls(config=cfg)

        task_id = task.get("task_id", "unknown")
        try:
            agent.initialize()
            start = datetime.now(timezone.utc)
            output = agent.run(task)
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            success = output is not None
            score = 1.0 if success else 0.0
            error = None
        except Exception as e:
            output = None
            elapsed = 0.0
            success = False
            score = 0.0
            error = str(e)

        cost_summary = agent.cost_summary() if hasattr(agent, "cost_summary") else {}
        total_cost = Decimal(str(cost_summary.get("total_cost_usd", "0")))
        total_input = int(cost_summary.get("total_input_tokens", 0))
        total_output = int(cost_summary.get("total_output_tokens", 0))
        avg_lat = float(cost_summary.get("avg_latency_ms", 0.0))

        agent.shutdown()

        result = CloudBaselineResult(
            provider=provider,
            model=model,
            benchmark=benchmark_name,
            task_id=task_id,
            success=success,
            score=score,
            cost_usd=total_cost,
            latency_ms=avg_lat,
            tokens_input=total_input,
            tokens_output=total_output,
            runtime_seconds=elapsed,
            error=error,
        )
        self._results.append(result)
        return result

    def run_benchmark(
        self,
        benchmark_name: str,
        tasks: list[dict[str, Any]],
        agents: list[tuple[str, str]],
    ) -> list[CloudBaselineResult]:
        results: list[CloudBaselineResult] = []
        for provider, model in agents:
            for task in tasks:
                r = self.run_single(provider, model, benchmark_name, task)
                results.append(r)
                logger.info(
                    "%s/%s on %s#%s: score=%.2f cost=$%.6f latency=%.1fms",
                    provider,
                    model,
                    benchmark_name,
                    task.get("task_id", "?"),
                    r.score,
                    float(r.cost_usd),
                    r.latency_ms,
                )
        return results

    def clear(self) -> None:
        self._results.clear()

    @property
    def results(self) -> list[CloudBaselineResult]:
        return list(self._results)
