"""
HTML report writer.

Purpose
-------
Render a self-contained, styled HTML report from an ``ExperimentSummary``.
The output includes all result tables, figure references, and statistical
analysis in a format suitable for sharing via browser or email.

Responsibilities
----------------
- Produce a single ``.html`` file with embedded CSS
- Render tables as HTML ``<table>`` elements
- Reference figure files via ``<img>`` tags
- Include a summary statistics section

Usage example
-------------
>>> from llm_reliability.reporting.html_report import HTMLReportWriter
>>> writer = HTMLReportWriter()
>>> html = writer.render(summary, figure_dir="figures")
>>> writer.save(summary, output_path="report.html", figure_dir="figures")

How reports are generated
-------------------------
``render()`` builds a complete HTML document using Python f-strings.
Table content is produced by calling ``pandas.DataFrame.to_html()`` with
appropriate CSS classes.  Figure paths are embedded as ``<img>`` elements.
"""

from __future__ import annotations

import html as html_escape_module
import json
import pathlib
from typing import Any

from llm_reliability.reporting.summary import ExperimentSummary
from llm_reliability.visualization.tables import TableGenerator


_CSS = """
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 14px;
    color: #1a1a2e;
    background: #f8f9fa;
    padding: 2rem;
  }
  .container { max-width: 1100px; margin: 0 auto; }
  h1 { font-size: 2rem; color: #0072B2; border-bottom: 3px solid #0072B2; padding-bottom: .5rem; margin-bottom: 1.5rem; }
  h2 { font-size: 1.4rem; color: #1a1a2e; margin: 2rem 0 .8rem; border-left: 4px solid #E69F00; padding-left: .6rem; }
  h3 { font-size: 1.1rem; color: #333; margin: 1.4rem 0 .4rem; }
  .meta { background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 2rem; }
  .meta table { border-collapse: collapse; width: 100%; }
  .meta td { padding: .3rem .8rem; }
  .meta td:first-child { font-weight: 600; color: #555; width: 180px; }
  .card { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08); padding: 1.5rem; margin-bottom: 1.5rem; }
  .result-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .result-table th { background: #0072B2; color: #fff; padding: .5rem .8rem; text-align: left; font-weight: 600; }
  .result-table tr:nth-child(even) { background: #f0f4ff; }
  .result-table td { padding: .4rem .8rem; border-bottom: 1px solid #e0e0e0; }
  .figure-grid { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }
  .figure-item { flex: 1 1 300px; text-align: center; }
  .figure-item img { max-width: 100%; border: 1px solid #dee2e6; border-radius: 6px; }
  .figure-item p { font-size: 12px; color: #666; margin-top: .3rem; font-style: italic; }
  .badge { display: inline-block; background: #0072B2; color: white; border-radius: 4px; padding: .1rem .5rem; font-size: 12px; font-weight: 600; }
  .badge.green { background: #009E73; }
  .badge.orange { background: #E69F00; }
  code { background: #eef2f7; padding: .1rem .4rem; border-radius: 4px; font-size: 12px; }
  pre { background: #1a1a2e; color: #e0e0ff; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 12px; }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1rem 0; }
  .stat-card { background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem; text-align: center; }
  .stat-card .value { font-size: 2rem; font-weight: 700; color: #0072B2; }
  .stat-card .label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: .05em; }
  footer { text-align: center; color: #aaa; font-size: 12px; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #dee2e6; }
</style>
"""


def _df_to_html(df: Any, cls: str = "result-table") -> str:
    """Convert a DataFrame to an HTML table string."""
    try:
        return df.to_html(classes=cls, index=False, border=0, escape=True)
    except Exception:
        return f"<pre>{html_escape_module.escape(str(df))}</pre>"


class HTMLReportWriter:
    """Renders a complete HTML research report from an ExperimentSummary.

    Parameters
    ----------
    table_gen : TableGenerator, optional
        Shared table generator.
    """

    def __init__(self, table_gen: TableGenerator | None = None) -> None:
        self._tg = table_gen or TableGenerator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        summary: ExperimentSummary,
        figure_dir: str | pathlib.Path = "figures",
    ) -> str:
        """Produce the full HTML report as a string.

        Parameters
        ----------
        summary : ExperimentSummary
        figure_dir : str | Path
            Relative path to figures.

        Returns
        -------
        str
            Complete HTML document.
        """
        figure_dir = pathlib.Path(figure_dir)
        body = "\n".join([
            self._header(summary),
            self._stats_overview(summary),
            self._section_results(summary, figure_dir),
            self._section_statistical_analysis(summary),
            self._section_discussion(summary),
            self._section_reproducibility(summary),
            f"<footer>Generated by LLM Reliability Ranking — {summary.generated_at}</footer>",
        ])

        return (
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
            f"<meta charset='UTF-8'>\n"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
            f"<title>{html_escape_module.escape(summary.experiment_name)}</title>\n"
            f"{_CSS}\n"
            "</head>\n<body>\n"
            f"<div class='container'>\n{body}\n</div>\n"
            "</body>\n</html>"
        )

    def save(
        self,
        summary: ExperimentSummary,
        output_path: str | pathlib.Path,
        figure_dir: str | pathlib.Path = "figures",
    ) -> pathlib.Path:
        """Render and write the report to *output_path*.

        Parameters
        ----------
        summary : ExperimentSummary
        output_path : str | Path
        figure_dir : str | Path

        Returns
        -------
        pathlib.Path
        """
        dest = pathlib.Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        html = self.render(summary, figure_dir=figure_dir)
        dest.write_text(html, encoding="utf-8")
        return dest

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _header(self, s: ExperimentSummary) -> str:
        return (
            f"<h1>{html_escape_module.escape(s.experiment_name)}</h1>\n"
            "<div class='meta'>\n"
            "<table>\n"
            f"<tr><td>Experiment ID</td><td><code>{s.experiment_id}</code></td></tr>\n"
            f"<tr><td>Generated At</td><td>{s.generated_at}</td></tr>\n"
            f"<tr><td>Benchmarks</td><td>{html_escape_module.escape(', '.join(s.benchmarks) or 'N/A')}</td></tr>\n"
            f"<tr><td>Agents</td><td>{html_escape_module.escape(', '.join(s.agents) or 'N/A')}</td></tr>\n"
            "</table>\n"
            "</div>"
        )

    def _stats_overview(self, s: ExperimentSummary) -> str:
        return (
            "<h2>Overview</h2>\n"
            "<div class='stat-grid'>\n"
            f"  <div class='stat-card'><div class='value'>{len(s.agents)}</div><div class='label'>Agents</div></div>\n"
            f"  <div class='stat-card'><div class='value'>{len(s.benchmarks)}</div><div class='label'>Benchmarks</div></div>\n"
            f"  <div class='stat-card'><div class='value'>{s.n_evaluations}</div><div class='label'>Evaluations</div></div>\n"
            f"  <div class='stat-card'><div class='value'>{s.n_executions}</div><div class='label'>Executions</div></div>\n"
            f"  <div class='stat-card'><div class='value'>{len(s.rankings)}</div><div class='label'>Rankings</div></div>\n"
            "</div>"
        )

    def _section_results(self, s: ExperimentSummary, figure_dir: pathlib.Path) -> str:
        parts = ["<h2>Results</h2>"]

        # Reliability metrics table
        if s.metrics:
            parts.append("<h3>Reliability Metrics</h3>")
            parts.append("<div class='card'>")
            try:
                df = self._tg.reliability_metrics_table(s.metrics)
                parts.append(_df_to_html(df))
            except Exception as exc:
                parts.append(f"<p><em>Table failed: {html_escape_module.escape(str(exc))}</em></p>")
            parts.append("</div>")

        # Ranking tables per benchmark
        for bench in s.benchmarks:
            s_rnks = [r for r in s.success_rankings if r.benchmark == bench]
            r_rnks = [r for r in s.reliability_rankings if r.benchmark == bench]
            if s_rnks and r_rnks:
                parts.append(f"<h3>Rankings — {html_escape_module.escape(bench)}</h3>")
                parts.append("<div class='card'>")
                try:
                    df = self._tg.ranking_table(s_rnks[0], r_rnks[0])
                    parts.append(_df_to_html(df))
                except Exception as exc:
                    parts.append(f"<p><em>Table failed: {html_escape_module.escape(str(exc))}</em></p>")
                parts.append("</div>")

        # Figures
        parts.append("<h3>Figures</h3>")
        parts.append("<div class='figure-grid'>")
        fig_refs = [
            ("ranking_comparison.png", "Success vs. Reliability Rankings"),
            ("rank_divergence.png", "Rank Divergence Scatter"),
            ("bump_chart.png", "Bump Chart"),
            ("scatter_success_vs_reliability.png", "Success vs. Reliability"),
            ("correlation_heatmap.png", "Correlation Heatmap"),
            ("box_plot.png", "Box Plot"),
            ("violin_plot.png", "Violin Plot"),
            ("score_histogram_success.png", "Success Rate Histogram"),
            ("score_histogram_reliability.png", "Reliability Histogram"),
        ]
        for fname, caption in fig_refs:
            fpath = figure_dir / fname
            parts.append(
                f"<div class='figure-item'>"
                f"<img src='{fpath}' alt='{html_escape_module.escape(caption)}' loading='lazy'>"
                f"<p>{html_escape_module.escape(caption)}</p>"
                f"</div>"
            )
        parts.append("</div>")

        return "\n".join(parts)

    def _section_statistical_analysis(self, s: ExperimentSummary) -> str:
        if not s.statistical_report:
            return "<h2>Statistical Analysis</h2><p><em>No statistical report available.</em></p>"

        parts = ["<h2>Statistical Analysis</h2>"]

        table_configs = [
            ("correlation_table", "Rank Correlations"),
            ("hypothesis_test_table", "Hypothesis Tests"),
            ("effect_size_table", "Effect Sizes"),
            ("confidence_interval_table", "Confidence Intervals"),
        ]
        for method, title in table_configs:
            parts.append(f"<h3>{title}</h3><div class='card'>")
            try:
                df = getattr(self._tg, method)(s.statistical_report)
                parts.append(_df_to_html(df))
            except Exception as exc:
                parts.append(f"<p><em>Failed: {html_escape_module.escape(str(exc))}</em></p>")
            parts.append("</div>")

        return "\n".join(parts)

    def _section_discussion(self, s: ExperimentSummary) -> str:
        return (
            "<h2>Discussion</h2>\n"
            "<div class='card'>\n"
            "<p>The central research question of this study is: <em>Under what conditions "
            "do success-based rankings diverge from reliability-based rankings of LLM agents?</em></p>"
            "<p>Success rate measures task completion in a single evaluation pass and is sensitive "
            "to favourable seed conditions or task-specific memorisation. "
            "Composite reliability, by contrast, aggregates evidence across repeated runs, "
            "perturbations, and fault-injection scenarios, producing a more conservative but "
            "more trustworthy ranking.</p>"
            "</div>"
        )

    def _section_reproducibility(self, s: ExperimentSummary) -> str:
        if not s.config_snapshot:
            return ""
        cfg_json = json.dumps(s.config_snapshot, indent=2, default=str)
        return (
            "<h2>Reproducibility</h2>\n"
            "<div class='card'>\n"
            f"<pre>{html_escape_module.escape(cfg_json)}</pre>\n"
            "</div>"
        )
