"""
Purpose
-------
Aggregate per-dimension reliability metrics into a single, configurable
composite ``ReliabilityScore`` for each agent and benchmark.

Responsibilities
----------------
- Accept per-agent and per-benchmark ScopeReliabilitySummary objects from
  ``ReliabilityMetricsEngine.compute_all()``.
- Apply configurable weights to the three reliability dimensions:
  consistency, robustness, and fault_tolerance.
- Produce ``ReliabilityScore`` objects per agent, per benchmark, and overall.

Usage example
-------------
>>> from llm_reliability.reliability.score_calculator import ReliabilityScoreCalculator
>>> calc = ReliabilityScoreCalculator(weights={"consistency": 0.5, "robustness": 0.3, "fault_tolerance": 0.2})
>>> scores = calc.compute(engine_output)
>>> for agent, score in scores.per_agent.items():
...     print(agent, score.composite_score)

Design notes
------------
When robustness or fault_tolerance data is absent (i.e., the underlying metric
returned a ``warning`` in its metadata), the calculator redistributes the weight
of missing dimensions equally across available dimensions rather than zeroing the
composite.  This matches the behaviour of ``ReliabilityMetricsEngine._compute_scope``.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight schema
# ---------------------------------------------------------------------------


class ReliabilityWeights(BaseModel):
    """Configurable weights for the four reliability dimensions.

    All weights must be non-negative and must sum to 1.0.
    """

    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency: float = Field(default=1 / 3, ge=0.0, le=1.0)
    robustness: float = Field(default=1 / 3, ge=0.0, le=1.0)
    fault_tolerance: float = Field(default=1 / 3, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> ReliabilityWeights:
        total = self.success_rate + self.consistency + self.robustness + self.fault_tolerance
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Reliability dimension weights must sum to 1.0, got {total:.6f}.")
        return self


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class ReliabilityScore(BaseModel):
    """Computed reliability score for a single scope (agent or benchmark).

    Attributes
    ----------
    scope_name : str
        Human-readable scope label (e.g. agent name or benchmark name).
    consistency_score : float
        Weighted contribution from repeated-run consistency.
    robustness_score : float
        Weighted contribution from prompt perturbation robustness.
    fault_tolerance_score : float
        Weighted contribution from fault tolerance.
    composite_score : float
        Final weighted aggregate in [0, 1].
    weights_used : ReliabilityWeights
        The actual weights applied (may differ from requested weights if
        dimensions were unavailable).
    available_dimensions : list[str]
        Dimensions for which data was available.
    notes : list[str]
        Informational messages about weight redistribution or missing data.
    """

    scope_name: str
    consistency_score: float = Field(ge=0.0, le=1.0)
    robustness_score: float = Field(ge=0.0, le=1.0)
    fault_tolerance_score: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)
    weights_used: ReliabilityWeights
    available_dimensions: list[str]
    notes: list[str] = Field(default_factory=list)


class ReliabilityScoreReport(BaseModel):
    """Container for all computed reliability scores.

    Attributes
    ----------
    per_agent : dict[str, ReliabilityScore]
        Score for each agent.
    per_benchmark : dict[str, ReliabilityScore]
        Score for each benchmark.
    overall : ReliabilityScore
        Macro-average score across all agents and benchmarks.
    """

    per_agent: dict[str, ReliabilityScore] = Field(default_factory=dict)
    per_benchmark: dict[str, ReliabilityScore] = Field(default_factory=dict)
    overall: ReliabilityScore | None = None

    def to_markdown(self) -> str:
        """Return a Markdown table summarising per-agent scores."""
        lines = [
            "# Reliability Score Report",
            "",
            "## Per-Agent Scores",
            "",
            "| Agent | Consistency | Robustness | Fault Tolerance | Composite |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for name, score in sorted(self.per_agent.items()):
            lines.append(
                f"| {name}"
                f" | {score.consistency_score:.4f}"
                f" | {score.robustness_score:.4f}"
                f" | {score.fault_tolerance_score:.4f}"
                f" | **{score.composite_score:.4f}** |"
            )

        if self.per_benchmark:
            lines.extend(
                [
                    "",
                    "## Per-Benchmark Scores",
                    "",
                    "| Benchmark | Consistency | Robustness | Fault Tolerance | Composite |",
                    "| :--- | :---: | :---: | :---: | :---: |",
                ]
            )
            for name, score in sorted(self.per_benchmark.items()):
                lines.append(
                    f"| {name}"
                    f" | {score.consistency_score:.4f}"
                    f" | {score.robustness_score:.4f}"
                    f" | {score.fault_tolerance_score:.4f}"
                    f" | **{score.composite_score:.4f}** |"
                )

        if self.overall:
            lines.extend(
                [
                    "",
                    "## Overall Score",
                    "",
                    f"- **Consistency**: {self.overall.consistency_score:.4f}",
                    f"- **Robustness**: {self.overall.robustness_score:.4f}",
                    f"- **Fault Tolerance**: {self.overall.fault_tolerance_score:.4f}",
                    f"- **Composite**: **{self.overall.composite_score:.4f}**",
                ]
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class ReliabilityScoreCalculator:
    """Compute weighted ``ReliabilityScore`` objects from engine output.

    Parameters
    ----------
    weights : ReliabilityWeights | dict | None
        Dimension weights. Accepts a ``ReliabilityWeights`` instance, a raw
        ``dict`` with keys ``consistency``, ``robustness``, ``fault_tolerance``,
        or ``None`` to use equal-weight defaults.
    """

    def __init__(
        self,
        weights: ReliabilityWeights | dict[str, float] | None = None,
    ) -> None:
        if weights is None:
            self._weights = ReliabilityWeights()
        elif isinstance(weights, dict):
            self._weights = ReliabilityWeights(**weights)
        else:
            self._weights = weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, engine_output: dict[str, Any]) -> ReliabilityScoreReport:
        """Compute reliability scores from ``ReliabilityMetricsEngine.compute_all()`` output.

        Parameters
        ----------
        engine_output : dict[str, Any]
            The dictionary returned by ``ReliabilityMetricsEngine.compute_all()``.

        Returns
        -------
        ReliabilityScoreReport
            Per-agent, per-benchmark, and overall reliability scores.
        """
        per_agent_raw: dict[str, Any] = engine_output.get("per_agent", {})
        per_bench_raw: dict[str, Any] = engine_output.get("per_benchmark", {})
        overall_raw: Any = engine_output.get("overall")

        per_agent: dict[str, ReliabilityScore] = {}
        for agent_name, summary in per_agent_raw.items():
            per_agent[agent_name] = self._score_from_summary(agent_name, summary)

        per_benchmark: dict[str, ReliabilityScore] = {}
        for bench_name, summary in per_bench_raw.items():
            per_benchmark[bench_name] = self._score_from_summary(bench_name, summary)

        overall: ReliabilityScore | None = None
        if overall_raw is not None:
            overall = self._score_from_summary("overall", overall_raw)

        return ReliabilityScoreReport(
            per_agent=per_agent,
            per_benchmark=per_benchmark,
            overall=overall,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_from_summary(
        self,
        scope_name: str,
        summary: Any,
    ) -> ReliabilityScore:
        """Compute a single ``ReliabilityScore`` from a ``ScopeReliabilitySummary``.

        Handles both dict-form (from ``to_dict()``) and live object forms.
        """
        # Support both native objects and dict representations
        if hasattr(summary, "consistency"):
            # Native ScopeReliabilitySummary object
            consistency_data = summary.consistency
            robustness_data = summary.robustness
            fault_data = summary.fault_tolerance
        else:
            # dict form (e.g., after serialisation)
            consistency_data = summary.get("consistency", {})
            robustness_data = summary.get("robustness", {})
            fault_data = summary.get("fault_tolerance", {})

        raw_c = self._extract_score(consistency_data, "deterministic_consistency_score")
        raw_r = self._extract_score(robustness_data, "robustness_score")
        raw_f = self._extract_score(fault_data, "fault_tolerance_score")

        r_available = self._is_available(robustness_data, "warning")
        f_available = self._is_available(fault_data, "warning")

        available: list[str] = ["consistency"]
        if r_available:
            available.append("robustness")
        if f_available:
            available.append("fault_tolerance")

        # Redistribute weights for unavailable dimensions
        effective_weights, notes = self._effective_weights(
            r_available=r_available,
            f_available=f_available,
        )

        composite = (
            effective_weights["consistency"] * raw_c
            + effective_weights["robustness"] * raw_r
            + effective_weights["fault_tolerance"] * raw_f
        )
        composite = min(max(composite, 0.0), 1.0)

        return ReliabilityScore(
            scope_name=scope_name,
            consistency_score=raw_c,
            robustness_score=raw_r,
            fault_tolerance_score=raw_f,
            composite_score=composite,
            weights_used=ReliabilityWeights(**effective_weights),
            available_dimensions=available,
            notes=notes,
        )

    @staticmethod
    def _extract_score(data: Any, field_name: str) -> float:
        """Extract a float score from an object or dict."""
        if data is None:
            return 0.0
        if hasattr(data, field_name):
            return float(getattr(data, field_name))
        if isinstance(data, dict):
            return float(data.get(field_name, 0.0))
        return 0.0

    @staticmethod
    def _is_available(data: Any, warning_key: str) -> bool:
        """Return True when the dimension has data (no warning flag)."""
        if data is None:
            return False
        if hasattr(data, "metadata"):
            return warning_key not in data.metadata
        if isinstance(data, dict):
            meta = data.get("metadata", {})
            return warning_key not in meta
        return True

    def _effective_weights(
        self,
        *,
        r_available: bool,
        f_available: bool,
    ) -> tuple[dict[str, float], list[str]]:
        """Redistribute weights for unavailable dimensions.

        If robustness or fault_tolerance are unavailable, their weight is
        divided equally among available dimensions.

        Returns
        -------
        tuple[dict[str, float], list[str]]
            Effective weight mapping and any informational notes.
        """
        notes: list[str] = []
        w_c = self._weights.consistency
        w_r = self._weights.robustness
        w_f = self._weights.fault_tolerance

        if not r_available:
            notes.append(
                "Robustness dimension unavailable; its weight redistributed to consistency and fault_tolerance."
            )
            redistribute = w_r / (1 if not f_available else 2)
            w_c += redistribute
            if f_available:
                w_f += redistribute
            w_r = 0.0

        if not f_available:
            notes.append(
                "Fault tolerance dimension unavailable; its weight redistributed to consistency."
            )
            # w_r already adjusted above if also unavailable
            w_c += w_f
            w_f = 0.0

        # Normalise to handle floating-point drift
        total = w_c + w_r + w_f
        if total > 0:
            w_c /= total
            w_r /= total
            w_f /= total

        return (
            {"consistency": w_c, "robustness": w_r, "fault_tolerance": w_f},
            notes,
        )
