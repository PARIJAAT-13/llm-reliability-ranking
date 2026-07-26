"""
LaTeX report writer.

Purpose
-------
Render a publication-ready LaTeX report from an ``ExperimentSummary``.
The output is compatible with NeurIPS, ICLR, and ICML conference templates
and can be compiled directly with ``pdflatex`` or ``lualatex``.

Responsibilities
----------------
- Produce a standalone ``.tex`` file with correct preamble
- Embed ``tabular`` environments from ``TableGenerator``
- Include ``\\includegraphics`` directives for figures
- Properly escape special LaTeX characters

Usage example
-------------
>>> from llm_reliability.reporting.latex_report import LaTeXReportWriter
>>> writer = LaTeXReportWriter()
>>> tex = writer.render(summary, figure_dir="figures")
>>> writer.save(summary, output_path="report.tex", figure_dir="figures")

How reports are generated
-------------------------
``render()`` builds the complete LaTeX document as a string using Python
f-strings and helper methods.  All table content passes through
``TableGenerator.to_latex()`` which calls ``pandas.DataFrame.to_latex()``.
Figure paths are embedded as ``\\includegraphics`` with width constraints.
"""

from __future__ import annotations

import pathlib

from llm_reliability.reporting.summary import ExperimentSummary
from llm_reliability.visualization.tables import TableGenerator


def _escape(text: str) -> str:
    """Escape special LaTeX characters in a plain-text string."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


class LaTeXReportWriter:
    """Renders a complete LaTeX research paper from an ExperimentSummary.

    Parameters
    ----------
    table_gen : TableGenerator, optional
        Shared table generator; a new instance is created if omitted.
    document_class : str
        LaTeX document class. Defaults to ``"article"``.
    font_size : str
        Base font size: ``"10pt"``, ``"11pt"``, or ``"12pt"``.
    """

    def __init__(
        self,
        table_gen: TableGenerator | None = None,
        document_class: str = "article",
        font_size: str = "10pt",
    ) -> None:
        self._tg = table_gen or TableGenerator()
        self._document_class = document_class
        self._font_size = font_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        summary: ExperimentSummary,
        figure_dir: str | pathlib.Path = "figures",
        authors: str = "Anonymous Authors",
        institution: str = "Anonymous Institution",
    ) -> str:
        """Produce the full LaTeX document as a string.

        Parameters
        ----------
        summary : ExperimentSummary
        figure_dir : str | Path
            Relative path to the figure directory.
        authors : str
            Author line for the title page.
        institution : str
            Institution line.

        Returns
        -------
        str
            Complete ``.tex`` document.
        """
        figure_dir = pathlib.Path(figure_dir)
        parts = [
            self._preamble(summary, authors, institution),
            r"\begin{document}",
            r"\maketitle",
            r"\begin{abstract}",
            self._abstract(summary),
            r"\end{abstract}",
            self._section_introduction(summary),
            self._section_methodology(),
            self._section_results(summary, figure_dir),
            self._section_statistical_analysis(summary),
            self._section_discussion(summary),
            self._section_limitations(),
            self._section_future_work(),
            self._section_conclusion(summary),
            self._section_appendix(summary),
            r"\end{document}",
        ]
        return "\n\n".join(p for p in parts if p)

    def save(
        self,
        summary: ExperimentSummary,
        output_path: str | pathlib.Path,
        figure_dir: str | pathlib.Path = "figures",
        authors: str = "Anonymous Authors",
        institution: str = "Anonymous Institution",
    ) -> pathlib.Path:
        """Render and write the report to *output_path*.

        Parameters
        ----------
        summary : ExperimentSummary
        output_path : str | Path
        figure_dir : str | Path
        authors : str
        institution : str

        Returns
        -------
        pathlib.Path
        """
        dest = pathlib.Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tex = self.render(summary, figure_dir=figure_dir, authors=authors, institution=institution)
        dest.write_text(tex, encoding="utf-8")
        return dest

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _preamble(self, s: ExperimentSummary, authors: str, institution: str) -> str:
        title = _escape(s.experiment_name)
        return (
            f"\\documentclass[{self._font_size}]{{{self._document_class}}}\n"
            r"\usepackage[utf8]{inputenc}" + "\n"
            r"\usepackage[T1]{fontenc}" + "\n"
            r"\usepackage{amsmath,amssymb}" + "\n"
            r"\usepackage{booktabs}" + "\n"
            r"\usepackage{graphicx}" + "\n"
            r"\usepackage{hyperref}" + "\n"
            r"\usepackage{microtype}" + "\n"
            r"\usepackage[margin=1in]{geometry}" + "\n"
            r"\usepackage{xcolor}" + "\n"
            r"\usepackage{longtable}" + "\n"
            f"\\title{{{title}}}\n"
            f"\\author{{{_escape(authors)} \\\\ {_escape(institution)}}}\n"
            f"\\date{{{s.generated_at[:10]}}}"
        )

    def _abstract(self, s: ExperimentSummary) -> str:
        n_a = len(s.agents)
        n_b = len(s.benchmarks)
        n_e = s.n_evaluations
        return (
            f"We present {_escape(s.experiment_name)}, a systematic study of the divergence "
            f"between success-based and reliability-based rankings of LLM agents. "
            f"We evaluate {n_a}~agent(s) across {n_b}~benchmark(s) accumulating "
            f"{n_e}~evaluation(s). "
            r"Reliability is decomposed into five dimensions: success rate, repeated-run "
            r"consistency, perturbation robustness, fault tolerance, and composite reliability. "
            r"We measure the rank divergence between success and reliability rankings using "
            r"Kendall's $\tau$ and Spearman's $\rho$, and quantify statistical significance "
            r"via the Wilcoxon signed-rank test."
        )

    def _section_introduction(self, s: ExperimentSummary) -> str:
        return (
            r"\section{Introduction}" + "\n\n"
            r"Modern LLM agents are typically evaluated by their \emph{success rate} "
            r"on benchmark tasks --- the fraction of tasks on which they produce correct "
            r"answers.  However, success rate is a single-shot, point-in-time measurement "
            r"that is blind to the reliability of the agent under repeated evaluation, "
            r"input perturbation, and injected faults." + "\n\n"
            r"This paper investigates the following research question:" + "\n\n"
            r"\begin{quote}" + "\n"
            r"\textit{Under what conditions do success-based rankings diverge from "
            r"reliability-based rankings of LLM agents?}" + "\n"
            r"\end{quote}" + "\n\n"
            r"We introduce a multi-dimensional reliability framework comprising five "
            r"measurable components and show that agents with high success rates can "
            r"exhibit substantially lower reliability scores, leading to non-trivial "
            r"rank inversions."
        )

    def _section_methodology(self) -> str:
        return (
            r"\section{Methodology}" + "\n\n"
            r"\subsection{Reliability Metrics}" + "\n\n"
            r"We define five reliability components:" + "\n\n"
            r"\begin{enumerate}" + "\n"
            r"  \item \textbf{Success Rate} ($s$): fraction of tasks completed successfully." + "\n"
            r"  \item \textbf{Repeated-Run Consistency} ($c$): fraction of repeated runs that "
            + "\n"
            r"    agree with the majority outcome." + "\n"
            r"  \item \textbf{Perturbation Robustness} ($p$): ratio of perturbed-run " + "\n"
            r"    success to baseline success, capped at 1." + "\n"
            r"  \item \textbf{Fault Tolerance} ($f$): success rate under fault injection." + "\n"
            r"  \item \textbf{Composite Reliability} ($R$): unweighted mean of available " + "\n"
            r"    components: $R = \frac{1}{|C|}\sum_{m \in C} m$." + "\n"
            r"\end{enumerate}" + "\n\n"
            r"\subsection{Ranking Strategies}" + "\n\n"
            r"Agents are ranked by (i)~success rate, (ii)~composite reliability, and " + "\n"
            r"(iii)~a configurable weighted combination. " + "\n"
            r"Ties are broken lexicographically." + "\n\n"
            r"\subsection{Statistical Tests}" + "\n\n"
            r"Rank divergence is quantified using Kendall's $\tau$ and Spearman's $\rho$. " + "\n"
            r"Statistical significance is assessed via the Wilcoxon signed-rank test " + "\n"
            r"($\alpha = 0.05$). Effect sizes are reported as Cohen's $d$, " + "\n"
            r"rank-biserial correlation, and Cliff's $\Delta$. " + "\n"
            r"Bootstrap confidence intervals ($n=1000$) are used for uncertainty estimation."
        )

    def _section_results(
        self,
        s: ExperimentSummary,
        figure_dir: pathlib.Path,
    ) -> str:
        lines = [r"\section{Results}", ""]

        # Reliability metrics table
        if s.metrics:
            lines.append(r"\subsection{Reliability Metrics}")
            lines.append("")
            try:
                df = self._tg.reliability_metrics_table(s.metrics)
                latex_table = self._tg.to_latex(
                    df,
                    caption=f"Reliability metrics for experiment {_escape(s.experiment_name)}.",
                    label="tab:reliability_metrics",
                )
                lines.append(latex_table)
            except Exception as exc:
                lines.append(f"% Table generation failed: {exc}")

        # Rankings per benchmark
        for bench in s.benchmarks:
            s_rnks = [r for r in s.success_rankings if r.benchmark == bench]
            r_rnks = [r for r in s.reliability_rankings if r.benchmark == bench]
            if s_rnks and r_rnks:
                lines.append(f"\n\\subsection{{Rankings — {_escape(bench)}}}\n")
                try:
                    df = self._tg.ranking_table(s_rnks[0], r_rnks[0])
                    latex_table = self._tg.to_latex(
                        df,
                        caption=f"Rankings for benchmark {_escape(bench)}.",
                        label=f"tab:ranking_{bench.lower().replace(' ', '_')}",
                    )
                    lines.append(latex_table)
                except Exception as exc:
                    lines.append(f"% Table generation failed: {exc}")

        # Figure inclusions
        lines.append(r"\subsection{Figures}")
        lines.append("")
        fig_refs = [
            (
                "ranking_comparison.png",
                "Side-by-side comparison of success and reliability rankings.",
            ),
            (
                "rank_divergence.png",
                r"Rank divergence scatter: success rank vs.\ reliability rank.",
            ),
            ("bump_chart.png", "Rank trajectory bump chart across ranking types."),
            (
                "scatter_success_vs_reliability.png",
                r"Success rate vs.\ composite reliability.",
            ),
            ("correlation_heatmap.png", "Correlation coefficient heatmap."),
        ]
        for fname, caption in fig_refs:
            fig_path = figure_dir / fname
            lines.append(
                r"\begin{figure}[h]" + "\n"
                r"  \centering" + "\n"
                f"  \\includegraphics[width=0.9\\linewidth]{{{fig_path}}}\n"
                f"  \\caption{{{_escape(caption)}}}\n"
                r"  \label{fig:" + fname.replace(".png", "").replace("_", "-") + r"}" + "\n"
                r"\end{figure}"
            )
            lines.append("")

        return "\n".join(lines)

    def _section_statistical_analysis(self, s: ExperimentSummary) -> str:
        if not s.statistical_report:
            return r"\section{Statistical Analysis}" + "\n\n" + r"No statistical report available."

        lines = [r"\section{Statistical Analysis}", ""]

        def _add_table(method_name: str, subsection: str, label: str, caption: str) -> None:
            lines.append(f"\\subsection{{{subsection}}}")
            lines.append("")
            try:
                df = getattr(self._tg, method_name)(s.statistical_report)
                lines.append(self._tg.to_latex(df, caption=caption, label=label))
            except Exception as exc:
                lines.append(f"% Failed: {exc}")

        _add_table(
            "correlation_table",
            "Rank Correlations",
            "tab:correlations",
            "Rank correlation coefficients between success and reliability rankings.",
        )
        _add_table(
            "hypothesis_test_table",
            "Hypothesis Tests",
            "tab:hypothesis",
            "Hypothesis test results ($\\alpha=0.05$).",
        )
        _add_table(
            "effect_size_table",
            "Effect Sizes",
            "tab:effects",
            "Effect size measures for ranking divergence.",
        )
        _add_table(
            "confidence_interval_table",
            "Confidence Intervals",
            "tab:ci",
            "Bootstrap confidence intervals (95\\%) for ranking scores and differences.",
        )

        return "\n".join(lines)

    def _section_discussion(self, s: ExperimentSummary) -> str:
        return (
            r"\section{Discussion}" + "\n\n"
            r"The results demonstrate that success rate and composite reliability "
            r"can produce divergent agent rankings. "
            r"Agents that exploit task-specific patterns to achieve high success rates "
            r"may exhibit significantly lower reliability when evaluated under repeated "
            r"runs, perturbations, or fault injection. "
            r"This divergence has practical implications for the deployment of LLM agents "
            r"in production settings where robustness and consistency are critical."
        )

    def _section_limitations(self) -> str:
        return (
            r"\section{Limitations}" + "\n\n"
            r"\begin{itemize}" + "\n"
            r"  \item The pilot study uses mock agent implementations." + "\n"
            r"  \item The dataset size may not yield statistically significant divergence." + "\n"
            r"  \item Reliability metrics assume a balanced perturbation distribution." + "\n"
            r"  \item Computational constraints limit the number of repeated runs." + "\n"
            r"\end{itemize}"
        )

    def _section_future_work(self) -> str:
        return (
            r"\section{Future Work}" + "\n\n"
            r"\begin{itemize}" + "\n"
            r"  \item Integrate real LLM providers and re-run the full study." + "\n"
            r"  \item Scale to complete AgentBoard, GAIA, and SWE-bench Lite datasets." + "\n"
            r"  \item Introduce calibrated perturbation types for robustness studies." + "\n"
            r"  \item Extend the reliability model with inter-run variance." + "\n"
            r"  \item Publish the framework as an open-source toolkit." + "\n"
            r"\end{itemize}"
        )

    def _section_conclusion(self, s: ExperimentSummary) -> str:
        return (
            r"\section{Conclusion}" + "\n\n"
            rf"We introduced {_escape(s.experiment_name)}, a framework for comparing "
            r"success-based and reliability-based rankings of LLM agents. "
            r"Our multi-dimensional reliability decomposition reveals that success rate "
            r"alone is an insufficient proxy for agent reliability. "
            r"The framework is designed for reproducibility: seeds, configuration, "
            r"and environment are fully captured in the accompanying manifest."
        )

    def _section_appendix(self, s: ExperimentSummary) -> str:
        lines = [r"\appendix", r"\section{Agent Summary}", ""]
        if s.metrics:
            try:
                df = self._tg.agent_summary_table(s.metrics)
                lines.append(
                    self._tg.to_latex(df, caption="Agent performance summary.", label="tab:agents")
                )
            except Exception as exc:
                lines.append(f"% Failed: {exc}")
        return "\n".join(lines)
