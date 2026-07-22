"""RankingRecord module.

This module defines the RankingRecord class, which represents the final rankings
generated from MetricRecords in the evaluation pipeline.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RankingRecord(BaseModel):
    """Immutable ranking record representing agent performance order.

    This record represents the final rankings of agents on a benchmark, before
    downstream statistical analysis is performed. Enforces validations on non-empty
    fields, positive ranks, and numeric scores.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    ranking_id: str
    benchmark: str
    ranking_name: str = Field(min_length=1)
    metric_ids: list[str]
    ranking_method: str = Field(min_length=1)
    rankings: dict[str, int]
    scores: dict[str, float]
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metric_ids")
    @classmethod
    def validate_metric_ids_non_empty(cls, v: list[str]) -> list[str]:
        """Verify that metric_ids is not empty."""
        if not v:
            raise ValueError("metric_ids cannot be empty")
        return v

    @field_validator("rankings")
    @classmethod
    def validate_rankings(cls, v: dict[str, int]) -> dict[str, int]:
        """Verify that rankings is not empty and all ranks are positive integers."""
        if not v:
            raise ValueError("rankings cannot be empty")
        for agent, rank in v.items():
            if rank <= 0:
                raise ValueError(f"rank for agent {agent} must be a positive integer, got {rank}")
        return v

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, v: dict[str, float]) -> dict[str, float]:
        """Verify that scores is not empty."""
        if not v:
            raise ValueError("scores cannot be empty")
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
    def from_canonical_json(cls, json_str: str) -> RankingRecord:
        """Deserialize a RankingRecord from a canonical JSON string.

        Args:
            json_str: The canonical JSON string representation of the record.

        Returns:
            An instance of RankingRecord.
        """
        return cls.model_validate(json.loads(json_str))
