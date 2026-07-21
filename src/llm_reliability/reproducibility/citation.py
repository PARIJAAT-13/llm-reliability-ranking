"""
CITATION.cff generator.

Purpose
-------
Produce a ``CITATION.cff`` file in the Citation File Format (CFF) v1.2 standard
so that researchers can cite the framework in their publications.

Responsibilities
----------------
- Generate valid CFF v1.2 YAML
- Include title, version, authors, repository, and experiment metadata
- Support ``cff-version: "1.2.0"`` schema

Usage example
-------------
>>> from llm_reliability.reproducibility.citation import CitationGenerator
>>> gen = CitationGenerator()
>>> cff = gen.build(experiment_name="My Experiment", version="0.1.0")
>>> gen.save(cff, "results/CITATION.cff")

How citations are generated
---------------------------
``build()`` produces a Python dictionary conforming to the CFF v1.2 schema,
then ``save()`` serialises it to YAML using the ``pyyaml`` library.  If
PyYAML is not installed, the output falls back to a hand-formatted YAML string
for robustness.
"""

from __future__ import annotations

import pathlib
from datetime import date
from typing import Any


_CFF_TEMPLATE = """\
cff-version: "1.2.0"
message: "If you use this software, please cite it as below."
type: software
title: "{title}"
version: "{version}"
date-released: "{date_released}"
abstract: "{abstract}"
authors:
  - family-names: "Anonymous"
    given-names: "Author"
    affiliation: "Anonymous Institution"
repository-code: "https://github.com/anonymous/llm-reliability-ranking"
license: MIT
keywords:
  - LLM
  - reliability
  - ranking
  - benchmark
  - agent
  - reproducibility
"""


class CitationGenerator:
    """Generates CITATION.cff files for research software citation."""

    def build(
        self,
        experiment_name: str,
        version: str = "0.1.0",
        abstract: str = (
            "A research framework for comparing success-based and "
            "reliability-based rankings of LLM agents."
        ),
        authors: list[dict[str, str]] | None = None,
        repository_url: str = "https://github.com/anonymous/llm-reliability-ranking",
        license_name: str = "MIT",
    ) -> dict[str, Any]:
        """Build a CFF v1.2 dictionary.

        Parameters
        ----------
        experiment_name : str
            Title of the software/experiment.
        version : str
            Software version string.
        abstract : str
            One-paragraph abstract describing the software.
        authors : list[dict[str, str]], optional
            List of author dicts with ``family-names``, ``given-names``, etc.
            Defaults to an anonymous placeholder.
        repository_url : str
            URL of the source code repository.
        license_name : str
            SPDX license identifier.

        Returns
        -------
        dict[str, Any]
            CFF-formatted dictionary.
        """
        if authors is None:
            authors = [
                {
                    "family-names": "Anonymous",
                    "given-names": "Author",
                    "affiliation": "Anonymous Institution",
                }
            ]

        return {
            "cff-version": "1.2.0",
            "message": "If you use this software, please cite it as below.",
            "type": "software",
            "title": experiment_name,
            "version": version,
            "date-released": str(date.today()),
            "abstract": abstract,
            "authors": authors,
            "repository-code": repository_url,
            "license": license_name,
            "keywords": [
                "LLM",
                "reliability",
                "ranking",
                "benchmark",
                "agent",
                "reproducibility",
            ],
        }

    def to_yaml(self, cff: dict[str, Any]) -> str:
        """Serialise a CFF dictionary to YAML.

        Parameters
        ----------
        cff : dict[str, Any]

        Returns
        -------
        str
            YAML-formatted CFF content.
        """
        try:
            import yaml
            return yaml.dump(cff, allow_unicode=True, default_flow_style=False, sort_keys=False)
        except ImportError:
            # Hand-format a valid YAML string without pyyaml
            return _CFF_TEMPLATE.format(
                title=cff.get("title", ""),
                version=cff.get("version", "0.1.0"),
                date_released=cff.get("date-released", str(date.today())),
                abstract=cff.get("abstract", ""),
            )

    def save(
        self,
        cff: dict[str, Any],
        path: str | pathlib.Path,
    ) -> pathlib.Path:
        """Write the CFF dictionary to *path*.

        Parameters
        ----------
        cff : dict[str, Any]
        path : str | Path

        Returns
        -------
        pathlib.Path
        """
        dest = pathlib.Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self.to_yaml(cff), encoding="utf-8")
        return dest
