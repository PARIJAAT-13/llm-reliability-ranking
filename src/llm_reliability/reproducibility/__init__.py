"""
Reproducibility package for the LLM Reliability Ranking framework.

Public API
----------
Capture
    ``EnvironmentCapture`` — snapshot Python version, packages, OS, git hash.

Manifest
    ``ManifestGenerator`` — produce ``manifest.json`` with record hashes.
    ``Manifest``           — Pydantic model for manifest data.

Citation
    ``CitationGenerator`` — produce ``CITATION.cff`` (CFF v1.2).

Checklist
    ``ReproducibilityChecklist`` — automated pass/fail reproducibility audit.
    ``ChecklistResult``          — audit result with Markdown rendering.

Archive
    ``ArchiveBuilder`` — assemble the complete results directory tree.
"""

from __future__ import annotations

from llm_reliability.reproducibility.archive import ArchiveBuilder
from llm_reliability.reproducibility.checklist import (
    CheckItem,
    ChecklistResult,
    ReproducibilityChecklist,
)
from llm_reliability.reproducibility.citation import CitationGenerator
from llm_reliability.reproducibility.environment import EnvironmentCapture
from llm_reliability.reproducibility.manifest import Manifest, ManifestGenerator

__all__ = [
    "EnvironmentCapture",
    "Manifest",
    "ManifestGenerator",
    "CitationGenerator",
    "ReproducibilityChecklist",
    "ChecklistResult",
    "CheckItem",
    "ArchiveBuilder",
]
