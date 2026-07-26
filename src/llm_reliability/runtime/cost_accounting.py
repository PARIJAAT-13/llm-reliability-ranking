"""Cost and token accounting for LLM inference calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CostEntry:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


PROVIDER_PRICING: dict[str, dict[str, tuple[float, float]]] = {
    "openai": {
        "gpt-4.1": (2.0, 8.0),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
        "gpt-4o": (2.50, 10.0),
        "gpt-4o-mini": (0.15, 0.60),
        "o3-mini": (1.10, 4.40),
        "o1": (15.0, 60.0),
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": (3.0, 15.0),
        "claude-3-5-haiku-20241022": (0.80, 4.0),
        "claude-3-opus-20240229": (15.0, 75.0),
        "claude-4-sonnet-20250514": (3.0, 15.0),
    },
    "google": {
        "gemini-2.5-pro-preview-03-25": (1.25, 5.0),
        "gemini-2.5-flash-preview-04-17": (0.15, 0.60),
        "gemini-2.0-flash": (0.10, 0.40),
    },
    "deepseek": {
        "deepseek-chat": (0.27, 1.10),
        "deepseek-reasoner": (0.55, 2.19),
    },
    "mistral": {
        "mistral-large-2407": (2.0, 6.0),
        "mistral-small-2409": (1.0, 3.0),
    },
    "cohere": {
        "command-r-plus": (2.5, 10.0),
        "command-r": (0.50, 1.50),
    },
}


class CostCalculator:
    """Calculates estimated costs for LLM API calls."""

    _pricing: dict[str, dict[str, tuple[float, float]]] = PROVIDER_PRICING

    @classmethod
    def register_pricing(
        cls, provider: str, model: str, input_price: float, output_price: float
    ) -> None:
        if provider not in cls._pricing:
            cls._pricing[provider] = {}
        cls._pricing[provider][model] = (input_price, output_price)

    @classmethod
    def estimate_cost(
        cls,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Decimal:
        provider_pricing = cls._pricing.get(provider, {})
        pricing = provider_pricing.get(model)
        if pricing is None:
            for pattern, prices in provider_pricing.items():
                if pattern in model or model.startswith(pattern):
                    pricing = prices
                    break
        if pricing is None:
            return Decimal("0.0")
        input_cost = Decimal(str(input_tokens / 1000.0)) * Decimal(str(pricing[0]))
        output_cost = Decimal(str(output_tokens / 1000.0)) * Decimal(str(pricing[1]))
        return input_cost + output_cost

    @classmethod
    def record_usage(
        cls,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        request_duration_ms: float = 0.0,
    ) -> CostEntry:
        cost = cls.estimate_cost(provider, model, input_tokens, output_tokens)
        return CostEntry(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            request_duration_ms=request_duration_ms,
        )


class CostTracker:
    """Tracks cumulative token usage and cost across multiple calls."""

    def __init__(self) -> None:
        self._entries: list[CostEntry] = []

    def add_entry(self, entry: CostEntry) -> None:
        self._entries.append(entry)

    def record_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        request_duration_ms: float = 0.0,
    ) -> CostEntry:
        entry = CostCalculator.record_usage(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            request_duration_ms=request_duration_ms,
        )
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[CostEntry]:
        return list(self._entries)

    @property
    def total_input_tokens(self) -> int:
        return sum(e.input_tokens for e in self._entries)

    @property
    def total_output_tokens(self) -> int:
        return sum(e.output_tokens for e in self._entries)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> Decimal:
        return sum((e.cost_usd for e in self._entries), Decimal("0.0"))

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def avg_latency_ms(self) -> float:
        if not self._entries:
            return 0.0
        return sum(e.latency_ms for e in self._entries) / len(self._entries)

    def summary(self) -> dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": float(self.total_cost_usd),
            "call_count": self.entry_count,
            "avg_latency_ms": self.avg_latency_ms,
        }

    def by_provider(self) -> dict[str, list[CostEntry]]:
        result: dict[str, list[CostEntry]] = {}
        for entry in self._entries:
            result.setdefault(entry.provider, []).append(entry)
        return result

    def by_model(self) -> dict[str, list[CostEntry]]:
        result: dict[str, list[CostEntry]] = {}
        for entry in self._entries:
            result.setdefault(entry.model, []).append(entry)
        return result

    def add(self, entry: CostEntry) -> None:
        """Backward-compatible alias for add_entry."""
        self.add_entry(entry)


# Backward-compatible alias
TokenAccount = CostTracker
