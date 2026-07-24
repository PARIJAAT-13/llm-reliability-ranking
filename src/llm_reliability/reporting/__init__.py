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

from llm_reliability.reporting.html_report import HTMLReportWriter
from llm_reliability.reporting.latex_report import LaTeXReportWriter
from llm_reliability.reporting.markdown_report import MarkdownReportWriter
from llm_reliability.reporting.report_generator import ReportGenerator
from llm_reliability.reporting.summary import ExperimentSummary

__all__ = [
    "ExperimentSummary",
    "ReportGenerator",
    "MarkdownReportWriter",
    "LaTeXReportWriter",
    "HTMLReportWriter",
]
