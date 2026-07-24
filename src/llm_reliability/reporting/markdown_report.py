"""
Markdown report writer.

Purpose
-------
Render a complete research report in GitHub-flavoured Markdown (GFM) from an
``ExperimentSummary``.  The report covers every section expected in an AI
conference paper: executive summary, methodology, results, statistical
analysis, discussion, limitations, future work, and appendix.

Responsibilities
----------------
- Produce a single self-contained ``.md`` file
- Embed tables rendered by ``TableGenerator``
- Reference figure files by relative path
- Include all statistical results in human-readable form

Usage example
-------------
>>> from llm_reliability.reporting.markdown_report import MarkdownReportWriter
>>> writer = MarkdownReportWriter()
>>> md = writer.render(summary, figure_dir="figures")
>>> writer.save(summary, output_path="report.md", figure_dir="figures")

How reports are generated
-------------------------
``render()`` calls a series of private ``_section_*`` methods, each of which
builds one section of the report as a string.  Sections are joined and written
to disk by ``save()``.  The ``TableGenerator`` (from ``visualization.tables``)
formats every tabular result.
"""

from __future__ import annotations

import pathlib

from llm_reliability.reporting.summary import ExperimentSummary
from llm_reliability.visualization.tables import TableGenerator


class MarkdownReportWriter:
    """Renders a complete Markdown research report from an ExperimentSummary.

    Parameters
    ----------
    table_gen : TableGenerator, optional
        Shared table generator; a new instance is created if omitted.
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
        """Produce the full Markdown report as a string.

        Parameters
        ----------
        summary : ExperimentSummary
            Aggregated experiment data.
        figure_dir : str | Path
            Relative path to figure directory (used in ``![...]`` links).

        Returns
        -------
        str
            Complete Markdown report.
        """
        figure_dir = pathlib.Path(figure_dir)
        sections = [
            self._section_header(summary),
            self._section_executive_summary(summary),
            self._section_overview(summary),
            self._section_methodology(summary),
            self._section_results(summary, figure_dir),
            self._section_statistical_analysis(summary),
            self._section_discussion(summary),
            self._section_limitations(),
            self._section_future_work(),
            self._section_appendix(summary),
        ]
        return "\n\n".join(s for s in sections if s)

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
            Destination file path.
        figure_dir : str | Path

        Returns
        -------
        pathlib.Path
            Resolved path of the written file.
        """
        dest = pathlib.Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        md = self.render(summary, figure_dir=figure_dir)
        dest.write_text(md, encoding="utf-8")
        return dest

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _section_header(self, s: ExperimentSummary) -> str:
        return (
            f"# {s.experiment_name}\n\n"
            f"**Experiment ID**: `{s.experiment_id}`  \n"
            f"**Generated**: {s.generated_at}  \n"
            f"**Framework**: LLM Reliability Ranking v0.1.0"
        )

    def _section_executive_summary(self, s: ExperimentSummary) -> str:
        n_agents = len(s.agents)
        n_benchmarks = len(s.benchmarks)
        n_evals = s.n_evaluations

        lines = [
            "## Executive Summary\n",
            f"This report presents the results of experiment **{s.experiment_name}**, "
            f"which evaluated **{n_agents} agent(s)** across **{n_benchmarks} benchmark(s)** "
            f"accumulating **{n_evals} evaluation(s)**.",
            "",
            "The central research question investigated is:",
            "",
            "> **Under what conditions do success-based rankings diverge from "
            "reliability-based rankings of LLM agents?**",
            "",
            "Key findings:",
        ]

        # Success vs reliability divergence
        bench_findings = []
        for bench in s.benchmarks:
            bench_findings.append(f"- Benchmark **{bench}**: {len(s.metrics_for_benchmark(bench))} agents evaluated.")

        lines.extend(bench_findings if bench_findings else ["- No benchmark data available."])

        if s.statistical_report:
            corrs = s.statistical_report.correlations
            if corrs:
                for name, cr in corrs.items():
                    coeff = cr.coefficient
                    interp = "strong" if abs(coeff) > 0.7 else ("moderate" if abs(coeff) > 0.4 else "weak")
                    lines.append(
                        f"- {name.replace('_', ' ').title()}: coefficient = {coeff:.4f} ({interp} correlation)."
                    )

        return "\n".join(lines)

    def _section_overview(self, s: ExperimentSummary) -> str:
        lines = [
            "## Experiment Overview\n",
            "| Property | Value |",
            "|---|---|",
            f"| Experiment ID | `{s.experiment_id}` |",
            f"| Experiment Name | {s.experiment_name} |",
            f"| Benchmarks | {', '.join(s.benchmarks) or 'N/A'} |",
            f"| Agents | {', '.join(s.agents) or 'N/A'} |",
            f"| Total Evaluations | {s.n_evaluations} |",
            f"| Total Executions | {s.n_executions} |",
            f"| Generated At | {s.generated_at} |",
        ]

        if s.metadata:
            lines.append("\n### Additional Metadata\n")
            for k, v in s.metadata.items():
                lines.append(f"- **{k}**: {v}")

        return "\n".join(lines)

    def _section_methodology(self, s: ExperimentSummary) -> str:
        return (
            "## Methodology\n\n"
            "### Reliability Metrics\n\n"
            "Five dimensions of reliability are measured for each agent:\n\n"
            "1. **Success Rate** — fraction of tasks completed successfully.\n"
            "2. **Repeated-Run Consistency** — fraction of runs that agree with "
            "   the majority outcome across repeated executions with identical inputs.\n"
            "3. **Perturbation Robustness** — ratio of perturbed-run success rate "
            "   to baseline success rate, capped at 1.0.\n"
            "4. **Fault Tolerance** — success rate under fault-injection conditions.\n"
            "5. **Composite Reliability** — unweighted arithmetic mean of available "
            "   component scores.\n\n"
            "### Ranking Strategies\n\n"
            "Three ranking strategies are compared:\n\n"
            "- **Success Ranking** — agents ordered by descending success rate.\n"
            "- **Reliability Ranking** — agents ordered by descending composite reliability.\n"
            "- **Weighted Ranking** — agents ordered by a configurable weighted combination "
            "  of metric dimensions.\n\n"
            "Ties are broken lexicographically by agent name to guarantee deterministic "
            "ordering across platforms and runs.\n\n"
            "### Statistical Analysis\n\n"
            "Ranking divergence is quantified using:\n\n"
            "- **Kendall's Tau** and **Spearman's rho** — rank correlation coefficients.\n"
            "- **Wilcoxon Signed-Rank Test** and **Paired t-test** — significance tests.\n"
            "- **Cohen's d**, **Rank-biserial correlation**, **Cliff's Delta** — effect sizes.\n"
            "- **Bootstrap confidence intervals** (n=1000, α=0.05) — uncertainty estimation."
        )

    def _section_results(
        self,
        s: ExperimentSummary,
        figure_dir: pathlib.Path,
    ) -> str:
        lines = ["## Results\n"]

        # Reliability metrics table
        if s.metrics:
            lines.append("### Reliability Metrics\n")
            try:
                df = self._tg.reliability_metrics_table(s.metrics)
                lines.append(self._tg.to_markdown(df))
            except Exception as exc:
                lines.append(f"*Table generation failed: {exc}*")

        # Ranking tables per benchmark
        for bench in s.benchmarks:
            s_ranks = [r for r in s.success_rankings if r.benchmark == bench]
            r_ranks = [r for r in s.reliability_rankings if r.benchmark == bench]

            if s_ranks and r_ranks:
                lines.append(f"\n### Rankings — {bench}\n")
                try:
                    df = self._tg.ranking_table(s_ranks[0], r_ranks[0])
                    lines.append(self._tg.to_markdown(df))
                except Exception as exc:
                    lines.append(f"*Table generation failed: {exc}*")

        # Figure references
        lines.append("\n### Figures\n")
        fig_refs = [
            ("ranking_bar_success.png", "Success Rankings (bar chart)"),
            ("ranking_bar_reliability.png", "Reliability Rankings (bar chart)"),
            ("ranking_comparison.png", "Success vs. Reliability Rankings"),
            ("rank_divergence.png", "Rank Divergence Scatter"),
            ("bump_chart.png", "Rank Trajectory Bump Chart"),
            ("score_histogram_success.png", "Success Rate Distribution"),
            ("score_histogram_reliability.png", "Reliability Score Distribution"),
            ("scatter_success_vs_reliability.png", "Success vs. Reliability Scatter"),
            ("correlation_heatmap.png", "Correlation Heatmap"),
            ("box_plot.png", "Metric Box Plot"),
            ("violin_plot.png", "Metric Violin Plot"),
        ]
        for fname, caption in fig_refs:
            ref_path = figure_dir / fname
            lines.append(f"![{caption}]({ref_path})\n*Figure: {caption}*\n")

        return "\n".join(lines)

    def _section_statistical_analysis(self, s: ExperimentSummary) -> str:
        if not s.statistical_report:
            return "## Statistical Analysis\n\n*No statistical report available.*"

        lines = ["## Statistical Analysis\n"]

        # Correlations
        lines.append("### Rank Correlations\n")
        try:
            df = self._tg.correlation_table(s.statistical_report)
            lines.append(self._tg.to_markdown(df))
        except Exception as exc:
            lines.append(f"*Failed: {exc}*")

        # Hypothesis tests
        lines.append("\n### Hypothesis Tests\n")
        try:
            df = self._tg.hypothesis_test_table(s.statistical_report)
            lines.append(self._tg.to_markdown(df))
        except Exception as exc:
            lines.append(f"*Failed: {exc}*")

        # Effect sizes
        lines.append("\n### Effect Sizes\n")
        try:
            df = self._tg.effect_size_table(s.statistical_report)
            lines.append(self._tg.to_markdown(df))
        except Exception as exc:
            lines.append(f"*Failed: {exc}*")

        # Confidence intervals
        lines.append("\n### Confidence Intervals\n")
        try:
            df = self._tg.confidence_interval_table(s.statistical_report)
            lines.append(self._tg.to_markdown(df))
        except Exception as exc:
            lines.append(f"*Failed: {exc}*")

        return "\n".join(lines)

    def _section_discussion(self, s: ExperimentSummary) -> str:
        lines = ["## Discussion\n"]

        # Auto-generate divergence commentary
        divergence_notes = []
        for bench in s.benchmarks:
            s_ranks = [r for r in s.success_rankings if r.benchmark == bench]
            r_ranks = [r for r in s.reliability_rankings if r.benchmark == bench]
            if s_ranks and r_ranks:
                s_map = dict(s_ranks[0].rank_map)
                r_map = dict(r_ranks[0].rank_map)
                divergent = [
                    (a, s_map[a], r_map[a])
                    for a in s_map
                    if a in r_map and abs(s_map[a] - r_map[a]) > 1
                ]
                if divergent:
                    for agent, sr, rr in divergent:
                        divergence_notes.append(
                            f"- **{agent}** on {bench}: success rank {sr}, "
                            f"reliability rank {rr} (Δ = {abs(sr - rr)})."
                        )

        if divergence_notes:
            lines.append(
                "Agents with a rank divergence > 1 between success and reliability rankings "
                "indicate conditions under which success-based evaluation is insufficient:\n"
            )
            lines.extend(divergence_notes)
            lines.append(
                "\nThese divergences suggest that high success rates can co-exist with "
                "low reliability when agents benefit from task-specific memorisation or "
                "favourable seed conditions rather than genuine generalisation."
            )
        else:
            lines.append(
                "No substantial rank divergence was observed in this pilot study. "
                "Larger-scale experiments with more agents and benchmarks are required "
                "to elicit the conditions under which success and reliability rankings diverge."
            )

        return "\n".join(lines)

    def _section_limitations(self) -> str:
        return (
            "## Limitations\n\n"
            "1. **Mock agents** — the pilot study uses mock agent implementations; "
            "   results should be replicated with actual LLM API calls.\n"
            "2. **Small dataset** — the current dataset size may not be sufficient "
            "   to detect statistically significant rank divergence.\n"
            "3. **Single benchmark** — cross-benchmark generalisation of the divergence "
            "   phenomenon requires evaluation on AgentBoard, GAIA, and SWE-bench Lite.\n"
            "4. **No calibration** — reliability metrics assume a balanced perturbation "
            "   distribution; calibrated perturbations may alter conclusions.\n"
            "5. **Computational budget** — repeated-run consistency estimation is limited "
            "   by the number of seeds; a minimum of 10 repetitions per condition is "
            "   recommended for robust estimates."
        )

    def _section_future_work(self) -> str:
        return (
            "## Future Work\n\n"
            "- Integrate real LLM providers (OpenAI GPT-4o, Google Gemini, Anthropic Claude, "
            "  DeepSeek) and re-run the full experimental study.\n"
            "- Scale to the complete AgentBoard, GAIA, and SWE-bench Lite datasets.\n"
            "- Introduce calibrated perturbation types (semantic paraphrase, format change, "
            "  noise injection) to study robustness under realistic distribution shift.\n"
            "- Extend the reliability model with inter-run variance as an additional "
            "  metric dimension.\n"
            "- Publish the framework as an open-source benchmark evaluation toolkit.\n"
            "- Conduct human evaluation to validate that the reliability-based ranking "
            "  correlates with practitioner judgement of agent trustworthiness."
        )

    def _section_appendix(self, s: ExperimentSummary) -> str:
        lines = ["## Appendix\n"]

        lines.append("### A. Agent Summary\n")
        if s.metrics:
            try:
                df = self._tg.agent_summary_table(s.metrics)
                lines.append(self._tg.to_markdown(df))
            except Exception as exc:
                lines.append(f"*Failed: {exc}*")
        else:
            lines.append("*No agent data available.*")

        lines.append("\n### B. Benchmark Summary\n")
        if s.metrics:
            try:
                df = self._tg.benchmark_summary_table(s.metrics)
                lines.append(self._tg.to_markdown(df))
            except Exception as exc:
                lines.append(f"*Failed: {exc}*")
        else:
            lines.append("*No benchmark data available.*")

        lines.append("\n### C. Configuration Snapshot\n")
        if s.config_snapshot:
            lines.append("```json")
            import json
            lines.append(json.dumps(s.config_snapshot, indent=2, default=str))
            lines.append("```")
        else:
            lines.append("*No configuration snapshot available.*")

        return "\n".join(lines)
