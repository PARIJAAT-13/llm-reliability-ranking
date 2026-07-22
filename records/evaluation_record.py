"""EvaluationRecord module.

This module defines the EvaluationRecord class, representing the benchmark's
evaluation of a single execution.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationRecord(BaseModel):
    """Immutable evaluation outcome derived from one execution.

    Stores the details of benchmark evaluations, ensuring immutability,
    validation of score limits, and support for canonical serialization and hashing.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    execution_id: str
    benchmark: str
    task_id: str
    success: bool
    score: float = Field(ge=0.0)
    max_score: float = Field(gt=0.0)
    passed: bool
    evaluation_time_seconds: float = Field(ge=0.0)
    evaluator_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)

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
    def from_canonical_json(cls, json_str: str) -> EvaluationRecord:
        """Deserialize an EvaluationRecord from a canonical JSON string.

        Args:
            json_str: The canonical JSON string representation of the record.

        Returns:
            An instance of EvaluationRecord.
        """
        return cls.model_validate(json.loads(json_str))
