"""
Reporting package for the LLM Reliability Ranking framework.

Public API
----------
Data
    ``ExperimentSummary`` — portable experiment context for all writers.

Writers
    ``MarkdownReportWriter`` — GitHub-flavoured Markdown report.
    ``LaTeXReportWriter``    — Conference-ready LaTeX paper.
    ``HTMLReportWriter``     — Self-contained HTML with embedded CSS.

Orchestrator
    ``ReportGenerator``      — Generates all formats in one call.
"""

from __future__ import annotations

from llm_reliability.reporting.html_report import HTMLReportWriter
from llm_reliability.reporting.latex_report import LaTeXReportWriter
from llm_reliability.reporting.markdown_report import MarkdownReportWriter
from llm_reliability.reporting.publication import (
    generate_benchmark_summary, generate_experiment_summary,
    generate_latex_table, generate_markdown_table, generate_ranking_summary,
    generate_reproducibility_manifest, generate_runtime_summary,
    generate_statistics_summary, save_publication_artifacts)
from llm_reliability.reporting.report_generator import ReportGenerator
from llm_reliability.reporting.summary import ExperimentSummary

__all__ = [
    "ExperimentSummary",
    "ReportGenerator",
    "MarkdownReportWriter",
    "LaTeXReportWriter",
    "HTMLReportWriter",
    "generate_experiment_summary",
    "generate_runtime_summary",
    "generate_benchmark_summary",
    "generate_ranking_summary",
    "generate_statistics_summary",
    "generate_latex_table",
    "generate_markdown_table",
    "generate_reproducibility_manifest",
    "save_publication_artifacts",
]
