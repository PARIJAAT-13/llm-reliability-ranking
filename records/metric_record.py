"""MetricRecord module.

This module defines the MetricRecord class, which stores derived metrics
computed from one or more EvaluationRecords.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MetricRecord(BaseModel):
    """Immutable metrics record computed from evaluation outcomes.

    Enforces ranges [0.0, 1.0] for metrics and verifies that evaluation_ids
    contains at least one identifier.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    metric_id: str
    evaluation_ids: list[str]
    agent: str
    benchmark: str
    task_id: str
    success_rate: float = Field(ge=0.0, le=1.0)
    repeated_run_consistency: float = Field(ge=0.0, le=1.0)
    perturbation_robustness: float = Field(ge=0.0, le=1.0)
    fault_tolerance: float = Field(ge=0.0, le=1.0)
    composite_reliability: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evaluation_ids")
    @classmethod
    def validate_evaluation_ids_non_empty(cls, v: list[str]) -> list[str]:
        """Verify that evaluation_ids list is not empty."""
        if not v:
            raise ValueError("evaluation_ids cannot be empty")
        return v

    def canonical_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation of the record.

        Excludes None values to remain consistent across different optional configurations.
        """
        return self.model_dump(mode="json", exclude_none=True)

    def canonical_json(self) -> str:
        """Return canonical JSON string with sorted keys and compact separators.

        Used for deterministic serialization.
        """
        return json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def sha256(self) -> str:
        """Return the SHA-256 hex digest of the canonical JSON representation."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_canonical_json(cls, json_str: str) -> MetricRecord:
        """Deserialize a MetricRecord from a canonical JSON string.

        Args:
            json_str: The canonical JSON string representation of the record.

        Returns:
            An instance of MetricRecord.
        """
        return cls.model_validate(json.loads(json_str))
