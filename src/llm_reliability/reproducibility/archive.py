"""
Archive builder — assembles the complete results directory tree.

Purpose
-------
Produce a self-contained, publication-ready archive directory from an
``ExperimentSummary`` by orchestrating all visualization, reporting, and
reproducibility subpackages.

Responsibilities
----------------
- Create the canonical results directory hierarchy
- Generate all figures (PNG, SVG, PDF)
- Export all tables (CSV, JSON, Markdown, LaTeX)
- Save all three report formats (Markdown, LaTeX, HTML)
- Write manifest.json, environment.json, CITATION.cff, CHECKLIST.md, README.md

Directory structure produced
-----------------------------
``results/<experiment_id>/``
├── figures/
│   ├── ranking_bar_success.{png,svg,pdf}
│   ├── ranking_bar_reliability.{png,svg,pdf}
│   ├── ranking_comparison.{png,svg,pdf}
│   ├── rank_divergence.{png,svg,pdf}
│   ├── bump_chart.{png,svg,pdf}
│   ├── score_histogram_success.{png,svg,pdf}
│   ├── score_histogram_reliability.{png,svg,pdf}
│   ├── scatter_success_vs_reliability.{png,svg,pdf}
│   ├── correlation_heatmap.{png,svg,pdf}
│   ├── box_plot.{png,svg,pdf}
│   └── violin_plot.{png,svg,pdf}
├── tables/
│   ├── reliability_metrics.{csv,json,md,tex}
│   ├── ranking.{csv,json,md,tex}
│   ├── hypothesis_tests.{csv,json,md,tex}
│   ├── effect_sizes.{csv,json,md,tex}
│   ├── confidence_intervals.{csv,json,md,tex}
│   └── agent_summary.{csv,json,md,tex}
├── reports/
│   ├── report.md
│   ├── report.tex
│   └── report.html
├── statistics/           (reserved for raw statistical outputs)
├── logs/                 (reserved for runner logs)
├── manifest.json
├── environment.json
├── CITATION.cff
├── CHECKLIST.md
└── README.md

Usage example
-------------
>>> from llm_reliability.reproducibility.archive import ArchiveBuilder
>>> builder = ArchiveBuilder()
>>> archive_dir = builder.build(summary, root_dir="results")
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)


class ArchiveBuilder:
    """Assembles the complete publication-ready experiment archive.

    Parameters
    ----------
    matplotlib_backend : str
        Matplotlib backend to use.  Set to ``"Agg"`` for headless environments.
    """

    def __init__(self, matplotlib_backend: str = "Agg") -> None:
        self._backend = matplotlib_backend
        self._set_backend()

    def _set_backend(self) -> None:
        import matplotlib

        matplotlib.use(self._backend, force=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        summary: Any,
        root_dir: str | pathlib.Path = "results",
        skip_excel: bool = True,
        formats: list[str] | None = None,
    ) -> pathlib.Path:
        """Build the complete archive under *root_dir/<experiment_id>*.

        Parameters
        ----------
        summary : ExperimentSummary
        root_dir : str | Path
            Parent directory for all experiments.
        skip_excel : bool
            Skip Excel export (avoids openpyxl dependency in CI).
        formats : list[str], optional
            Which report formats to generate; defaults to all three.

        Returns
        -------
        pathlib.Path
            Path to the experiment archive directory.
        """
        exp_dir = pathlib.Path(root_dir) / summary.experiment_id
        figures_dir = exp_dir / "figures"
        tables_dir = exp_dir / "tables"
        reports_dir = exp_dir / "reports"
        statistics_dir = exp_dir / "statistics"
        logs_dir = exp_dir / "logs"

        for d in [figures_dir, tables_dir, reports_dir, statistics_dir, logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        logger.info("Building archive at %s", exp_dir)

        self._generate_figures(summary, figures_dir)
        self._generate_tables(summary, tables_dir, skip_excel=skip_excel)
        self._generate_reports(summary, reports_dir, figures_dir, formats=formats)
        self._write_manifest(summary, exp_dir)
        self._write_environment(exp_dir)
        self._write_citation(summary, exp_dir)
        self._write_checklist(summary, exp_dir)
        self._write_readme(summary, exp_dir)

        logger.info("Archive complete: %s", exp_dir)
        return exp_dir

    # ------------------------------------------------------------------
    # Figure generation
    # ------------------------------------------------------------------

    def _generate_figures(self, summary: Any, figures_dir: pathlib.Path) -> None:
        """Produce all visualisation figures."""
        try:
            from llm_reliability.visualization.distributions import DistributionPlotter
            from llm_reliability.visualization.export import FigureExporter
            from llm_reliability.visualization.heatmaps import HeatmapPlotter
            from llm_reliability.visualization.ranking_plots import RankingPlotter
        except ImportError as exc:
            logger.warning("Visualization imports failed: %s", exc)
            return

        rp = RankingPlotter()
        dp = DistributionPlotter()
        hp = HeatmapPlotter()

        metrics = summary.metrics
        s_rankings = summary.success_rankings
        r_rankings = summary.reliability_rankings

        # Success ranking bar
        if s_rankings:
            try:
                fig = rp.plot_ranking_bar(s_rankings[0], title="Success Rankings")
                FigureExporter.save_all(fig, figures_dir / "ranking_bar_success")
            except Exception as exc:
                logger.warning("ranking_bar_success failed: %s", exc)

        # Reliability ranking bar
        if r_rankings:
            try:
                fig = rp.plot_ranking_bar(r_rankings[0], title="Reliability Rankings")
                FigureExporter.save_all(fig, figures_dir / "ranking_bar_reliability")
            except Exception as exc:
                logger.warning("ranking_bar_reliability failed: %s", exc)

        # Comparison and divergence
        if s_rankings and r_rankings:
            try:
                fig = rp.plot_ranking_comparison(s_rankings[0], r_rankings[0])
                FigureExporter.save_all(fig, figures_dir / "ranking_comparison")
            except Exception as exc:
                logger.warning("ranking_comparison failed: %s", exc)

            try:
                fig = rp.plot_rank_divergence(s_rankings[0], r_rankings[0])
                FigureExporter.save_all(fig, figures_dir / "rank_divergence")
            except Exception as exc:
                logger.warning("rank_divergence failed: %s", exc)

        # Bump chart
        if summary.rankings:
            try:
                fig = rp.plot_bump_chart(summary.rankings[:3])
                FigureExporter.save_all(fig, figures_dir / "bump_chart")
            except Exception as exc:
                logger.warning("bump_chart failed: %s", exc)

        # Distribution plots
        if metrics:
            for metric_name, stem in [
                ("success_rate", "score_histogram_success"),
                ("composite_reliability", "score_histogram_reliability"),
            ]:
                try:
                    fig = dp.plot_score_histogram(metrics, metric=metric_name)
                    FigureExporter.save_all(fig, figures_dir / stem)
                except Exception as exc:
                    logger.warning("%s failed: %s", stem, exc)

            try:
                fig = dp.plot_scatter_success_vs_reliability(metrics)
                FigureExporter.save_all(fig, figures_dir / "scatter_success_vs_reliability")
            except Exception as exc:
                logger.warning("scatter failed: %s", exc)

            try:
                fig = dp.plot_box(metrics)
                FigureExporter.save_all(fig, figures_dir / "box_plot")
            except Exception as exc:
                logger.warning("box_plot failed: %s", exc)

            try:
                fig = dp.plot_violin(metrics)
                FigureExporter.save_all(fig, figures_dir / "violin_plot")
            except Exception as exc:
                logger.warning("violin_plot failed: %s", exc)

        # Heatmap
        if summary.statistical_report and summary.statistical_report.correlations:
            try:
                fig = hp.plot_from_dict(
                    summary.statistical_report.correlations,
                    title="Rank Correlation Heatmap",
                )
                FigureExporter.save_all(fig, figures_dir / "correlation_heatmap")
            except Exception as exc:
                logger.warning("correlation_heatmap failed: %s", exc)

    # ------------------------------------------------------------------
    # Table generation
    # ------------------------------------------------------------------

    def _generate_tables(
        self,
        summary: Any,
        tables_dir: pathlib.Path,
        skip_excel: bool = True,
    ) -> None:
        """Export all data tables."""
        try:
            from llm_reliability.visualization.export import TableExporter
            from llm_reliability.visualization.tables import TableGenerator
        except ImportError as exc:
            logger.warning("Table imports failed: %s", exc)
            return

        tg = TableGenerator()
        metrics = summary.metrics
        s_rnks = summary.success_rankings
        r_rnks = summary.reliability_rankings
        stat_report = summary.statistical_report

        # Reliability metrics
        if metrics:
            try:
                df = tg.reliability_metrics_table(metrics)
                TableExporter.save_all(
                    df,
                    tables_dir / "reliability_metrics",
                    caption="Reliability metrics",
                    skip_excel=skip_excel,
                )
            except Exception as exc:
                logger.warning("reliability_metrics table failed: %s", exc)

            try:
                df = tg.agent_summary_table(metrics)
                TableExporter.save_all(
                    df, tables_dir / "agent_summary", caption="Agent summary", skip_excel=skip_excel
                )
            except Exception as exc:
                logger.warning("agent_summary table failed: %s", exc)

            try:
                df = tg.benchmark_summary_table(metrics)
                TableExporter.save_all(
                    df,
                    tables_dir / "benchmark_summary",
                    caption="Benchmark summary",
                    skip_excel=skip_excel,
                )
            except Exception as exc:
                logger.warning("benchmark_summary table failed: %s", exc)

        # Ranking table
        if s_rnks and r_rnks:
            try:
                df = tg.ranking_table(s_rnks[0], r_rnks[0])
                TableExporter.save_all(
                    df, tables_dir / "ranking", caption="Ranking comparison", skip_excel=skip_excel
                )
            except Exception as exc:
                logger.warning("ranking table failed: %s", exc)

        # Statistical tables
        if stat_report:
            for method_name, stem, caption in [
                ("hypothesis_test_table", "hypothesis_tests", "Hypothesis test results"),
                ("effect_size_table", "effect_sizes", "Effect sizes"),
                ("confidence_interval_table", "confidence_intervals", "Confidence intervals"),
                ("correlation_table", "correlations", "Rank correlations"),
            ]:
                try:
                    df = getattr(tg, method_name)(stat_report)
                    TableExporter.save_all(
                        df, tables_dir / stem, caption=caption, skip_excel=skip_excel
                    )
                except Exception as exc:
                    logger.warning("%s table failed: %s", stem, exc)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_reports(
        self,
        summary: Any,
        reports_dir: pathlib.Path,
        figures_dir: pathlib.Path,
        formats: list[str] | None = None,
    ) -> None:
        """Generate all report formats."""
        try:
            from llm_reliability.reporting.report_generator import ReportGenerator
        except ImportError as exc:
            logger.warning("Reporting imports failed: %s", exc)
            return

        rel_figures = pathlib.Path("../figures")
        gen = ReportGenerator()
        try:
            gen.generate(
                summary,
                output_dir=reports_dir,
                formats=formats,  # type: ignore[arg-type]
                figure_dir=rel_figures,
            )
        except Exception as exc:
            logger.warning("Report generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Reproducibility files
    # ------------------------------------------------------------------

    def _write_manifest(self, summary: Any, exp_dir: pathlib.Path) -> None:
        try:
            from llm_reliability.reproducibility.environment import EnvironmentCapture
            from llm_reliability.reproducibility.manifest import ManifestGenerator

            env = EnvironmentCapture.capture()
            gen = ManifestGenerator()
            manifest = gen.build(summary, environment=env)
            gen.save(manifest, exp_dir / "manifest.json")
        except Exception as exc:
            logger.warning("manifest.json generation failed: %s", exc)

    def _write_environment(self, exp_dir: pathlib.Path) -> None:
        try:
            from llm_reliability.reproducibility.environment import EnvironmentCapture

            env = EnvironmentCapture.capture()
            dest = exp_dir / "environment.json"
            dest.write_text(json.dumps(env.to_dict(), indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.warning("environment.json generation failed: %s", exc)

    def _write_citation(self, summary: Any, exp_dir: pathlib.Path) -> None:
        try:
            from llm_reliability.reproducibility.citation import CitationGenerator

            gen = CitationGenerator()
            cff = gen.build(experiment_name=summary.experiment_name)
            gen.save(cff, exp_dir / "CITATION.cff")
        except Exception as exc:
            logger.warning("CITATION.cff generation failed: %s", exc)

    def _write_checklist(self, summary: Any, exp_dir: pathlib.Path) -> None:
        try:
            from llm_reliability.reproducibility.checklist import (
                ReproducibilityChecklist,
            )

            checker = ReproducibilityChecklist()
            result = checker.run(summary, archive_dir=exp_dir)
            checker.save(result, exp_dir / "CHECKLIST.md")
        except Exception as exc:
            logger.warning("CHECKLIST.md generation failed: %s", exc)

    def _write_readme(self, summary: Any, exp_dir: pathlib.Path) -> None:
        """Write a README.md for the archive."""
        readme = f"""# Experiment Archive: {summary.experiment_name}

**Experiment ID**: `{summary.experiment_id}`
**Generated**: {summary.generated_at}

## Directory Structure

```
{summary.experiment_id}/
├── figures/       — Publication-quality figures (PNG, SVG, PDF)
├── tables/        — Result tables (CSV, JSON, Markdown, LaTeX)
├── reports/       — Full reports (Markdown, LaTeX, HTML)
├── statistics/    — Raw statistical outputs
├── logs/          — Experiment runner logs
├── manifest.json  — Reproducibility manifest (record hashes, seeds, git commit)
├── environment.json — Execution environment snapshot
├── CITATION.cff   — CFF v1.2 citation file
├── CHECKLIST.md   — Reproducibility checklist
└── README.md      — This file
```

## Research Question

> Under what conditions do success-based rankings diverge from
> reliability-based rankings of LLM agents?

## Agents

{chr(10).join(f"- {a}" for a in summary.agents) or "N/A"}

## Benchmarks

{chr(10).join(f"- {b}" for b in summary.benchmarks) or "N/A"}

## Statistics

- **Total Evaluations**: {summary.n_evaluations}
- **Total Executions**: {summary.n_executions}
- **Ranking Records**: {len(summary.rankings)}

## Citation

Please cite this work using the `CITATION.cff` file in this directory.

## Reproducibility

See `manifest.json` for record hashes, seeds, and git commit hash.
See `CHECKLIST.md` for automated reproducibility audit results.
"""
        dest = exp_dir / "README.md"
        dest.write_text(readme, encoding="utf-8")
