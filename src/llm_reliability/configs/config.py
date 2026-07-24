"""
Purpose
-------
Define the immutable, versioned experiment configuration that drives every
pipeline run and anchors reproducibility via deterministic hashing.

Responsibilities
----------------
- Validate all experiment parameters at construction time
- Reject unknown fields to prevent silent misconfiguration
- Provide canonical serialization and SHA-256 content hashing
- Support equality comparison and schema versioning

Usage example
-------------
>>> from llm_reliability.configs import Configuration
>>> cfg = Configuration(
...     experiment_name="pilot",
...     benchmark="agentboard",
...     agent="mock_agent",
...     llm="gpt-4",
...     prompt_version="v1",
...     dataset_version="1.0",
...     seed=42,
...     repetitions=5,
... )
>>> cfg.sha256()
'...'

Design notes
------------
Configuration is the root artifact in the dependency graph. Every downstream
record stores ``configuration_hash`` rather than embedding the full object,
which keeps records compact while preserving reproducibility links.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from llm_reliability.utils.serialization import SerializableModel

CONFIG_VERSION = "1.1.0"


class ReliabilityWeightsConfig(SerializableModel):
    """Configurable weights for all four reliability dimensions.

    All weights must be non-negative and sum to 1.0.
    """

    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency: float = Field(default=1 / 3, ge=0.0, le=1.0)
    robustness: float = Field(default=1 / 3, ge=0.0, le=1.0)
    fault_tolerance: float = Field(default=1 / 3, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_sum(self) -> ReliabilityWeightsConfig:
        total = self.success_rate + self.consistency + self.robustness + self.fault_tolerance
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"reliability_weights must sum to 1.0, got {total:.6f}.")
        return self


class VisualizationOptions(SerializableModel):
    """Optional settings for the visualization pipeline."""

    output_dir: str = Field(default="results/figures")
    formats: list[str] = Field(default_factory=lambda: ["png"])
    dpi: int = Field(default=150, ge=72, le=600)
    style: str = Field(default="dark")
    interactive: bool = Field(default=False)


class StatisticalOptions(SerializableModel):
    """Optional settings for the statistical analysis pipeline."""

    confidence_level: float = Field(default=0.95, ge=0.5, le=0.999)
    bootstrap_iterations: int = Field(default=1000, ge=100)
    alpha: float = Field(default=0.05, ge=0.001, le=0.5)
    compute_divergence: bool = Field(default=True)


class Configuration(SerializableModel):
    """Immutable, versioned experiment configuration."""

    version: str = Field(
        default=CONFIG_VERSION,
        description="Configuration schema version in semver format.",
    )
    experiment_name: str = Field(min_length=1)
    benchmark: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    llm: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    seed: int = Field(ge=0)
    repetitions: int = Field(gt=0)
    perturbations: tuple[str, ...] = Field(default_factory=tuple)
    fault_injection: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    # -----------------------------------------------------------------------
    # Optional extension fields (Part 9) — all have defaults; no breaking change
    # -----------------------------------------------------------------------
    reliability_weights: ReliabilityWeightsConfig = Field(
        default_factory=ReliabilityWeightsConfig,
        description="Configurable weights for the three reliability dimensions.",
    )
    visualization: VisualizationOptions = Field(
        default_factory=VisualizationOptions,
        description="Settings passed to the visualization pipeline.",
    )
    statistical: StatisticalOptions = Field(
        default_factory=StatisticalOptions,
        description="Settings passed to the statistical analysis pipeline.",
    )

    @field_validator("version")
    @classmethod
    def validate_version_format(cls, value: str) -> str:
        """Ensure version follows major.minor.patch semver."""
        parts = value.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            msg = f"version must be semver major.minor.patch, got {value!r}"
            raise ValueError(msg)
        return value

    @classmethod
    def from_file(cls, path: Path | str) -> Configuration:
        """Load configuration from a UTF-8 JSON file."""
        file_path = Path(path)
        return cls.from_canonical_json(file_path.read_text(encoding="utf-8"))

    def write_file(self, path: Path | str) -> None:
        """Write canonical JSON to a UTF-8 file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(self.canonical_json(), encoding="utf-8")
