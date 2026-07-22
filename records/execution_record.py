"""ExecutionRecord module.

This module defines the ExecutionRecord class, representing the raw execution details
produced when an Agent executes a benchmark task.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExecutionRecord(BaseModel):
    """Immutable record of an agent execution on a benchmark task.

    Captures the raw results of execution (telemetry, stdout, stderr, outputs)
    and enforces strict validation.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    execution_id: str
    configuration_hash: str
    benchmark: str
    agent: str
    task_id: str
    status: Literal["SUCCESS", "FAILED", "TIMEOUT", "ERROR"]
    start_time: datetime
    end_time: datetime
    runtime_seconds: float = Field(ge=0.0)
    stdout: str
    stderr: str
    error_message: str | None = None
    raw_output: dict[str, Any] = Field(default_factory=dict)
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
    def from_canonical_json(cls, json_str: str) -> ExecutionRecord:
        """Deserialize an ExecutionRecord from a canonical JSON string.

        Args:
            json_str: The canonical JSON string representation of the record.

        Returns:
            An instance of ExecutionRecord.
        """
        return cls.model_validate(json.loads(json_str))
