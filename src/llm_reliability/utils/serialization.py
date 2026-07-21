"""
Purpose
-------
Provide canonical serialization, deserialization, and hashing for immutable
framework models and records.

Responsibilities
----------------
- Deterministic JSON encoding with sorted keys
- SHA-256 content hashing
- Round-trip deserialization via Pydantic validation

Usage example
-------------
>>> from llm_reliability.utils.serialization import SerializableModel
>>> class MyRecord(SerializableModel):
...     value: int
>>> record = MyRecord(value=1)
>>> restored = MyRecord.from_canonical_json(record.canonical_json())
>>> record == restored
True

Design notes
------------
All records and configuration objects inherit from ``SerializableModel`` to
guarantee a single, reproducible serialization contract across the pipeline.
Hashing operates on canonical JSON bytes so identical semantic content always
produces identical digests regardless of construction order.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

SerializableModelT = TypeVar("SerializableModelT", bound="SerializableModel")


class SerializableModel(BaseModel):
    """Base class for immutable, canonically serializable framework objects."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    def canonical_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary suitable for hashing."""
        return self.model_dump(mode="json", exclude_none=True)

    def canonical_json(self) -> str:
        """Return canonical JSON with sorted keys and compact separators."""
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
    def from_canonical_json(
        cls: type[SerializableModelT],
        json_str: str,
    ) -> SerializableModelT:
        """Deserialize from canonical JSON via Pydantic validation."""
        return cls.model_validate(json.loads(json_str))
