"""
ReportGenerator — orchestrator for all three report formats.

Purpose
-------
Provide a single entry point that generates Markdown, LaTeX, and HTML reports
simultaneously from an ``ExperimentSummary`` and an output directory.

Responsibilities
----------------
- Coordinate ``MarkdownReportWriter``, ``LaTeXReportWriter``, ``HTMLReportWriter``
- Accept a format list so callers can request specific output types
- Return paths to all generated files

Usage example
-------------
>>> from llm_reliability.reporting.report_generator import ReportGenerator
>>> gen = ReportGenerator()
>>> paths = gen.generate(summary, output_dir="results/exp-001/reports")

How reports are generated
-------------------------
``generate()`` instantiates each requested writer and calls its ``save()``
method.  A ``figure_dir`` is computed relative to the output directory so
embedded figure paths resolve correctly when the report is opened in place.
"""

from __future__ import annotations

import pathlib
from typing import Literal

from llm_reliability.reporting.html_report import HTMLReportWriter
from llm_reliability.reporting.latex_report import LaTeXReportWriter
from llm_reliability.reporting.markdown_report import MarkdownReportWriter
from llm_reliability.reporting.summary import ExperimentSummary
from llm_reliability.visualization.tables import TableGenerator

ReportFormat = Literal["markdown", "latex", "html"]


class ReportGenerator:
    """Orchestrates all report format writers.

    Parameters
    ----------
    table_gen : TableGenerator, optional
        Shared table generator passed to all writers.
    """

    def __init__(self, table_gen: TableGenerator | None = None) -> None:
        _tg = table_gen or TableGenerator()
        self._md_writer = MarkdownReportWriter(table_gen=_tg)
        self._latex_writer = LaTeXReportWriter(table_gen=_tg)
        self._html_writer = HTMLReportWriter(table_gen=_tg)

    def generate(
        self,
        summary: ExperimentSummary,
        output_dir: str | pathlib.Path,
        formats: list[ReportFormat] | None = None,
        figure_dir: str | pathlib.Path | None = None,
        latex_authors: str = "Anonymous Authors",
        latex_institution: str = "Anonymous Institution",
    ) -> dict[str, pathlib.Path]:
        """Generate reports in the requested formats.

        Parameters
        ----------
        summary : ExperimentSummary
            Aggregated experiment data.
        output_dir : str | Path
            Directory where reports are written.
        formats : list[ReportFormat], optional
            Which formats to produce.  Defaults to all three.
        figure_dir : str | Path, optional
            Path to figures (relative to *output_dir*).
            Defaults to ``"../figures"`` (sibling of reports/).
        latex_authors : str
            Author line for the LaTeX title page.
        latex_institution : str
            Institution line for LaTeX.

        Returns
        -------
        dict[str, pathlib.Path]
            Mapping of format name → output file path.
        """
        if formats is None:
            formats = ["markdown", "latex", "html"]

        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if figure_dir is None:
            figure_dir = pathlib.Path("../figures")
        else:
            figure_dir = pathlib.Path(figure_dir)

        results: dict[str, pathlib.Path] = {}

        if "markdown" in formats:
            dest = output_dir / "report.md"
            results["markdown"] = self._md_writer.save(
                summary, output_path=dest, figure_dir=figure_dir
            )

        if "latex" in formats:
            dest = output_dir / "report.tex"
            results["latex"] = self._latex_writer.save(
                summary,
                output_path=dest,
                figure_dir=figure_dir,
                authors=latex_authors,
                institution=latex_institution,
            )

        if "html" in formats:
            dest = output_dir / "report.html"
            results["html"] = self._html_writer.save(
                summary, output_path=dest, figure_dir=figure_dir
            )

        return results
