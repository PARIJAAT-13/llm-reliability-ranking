#!/usr/bin/env python3
"""
generate_report.py — CLI entry point for the Visualization, Reporting and
Publication Package.

Usage
-----
    python scripts/generate_report.py \\
        --experiment-dir results/<experiment_id> \\
        --output-dir results/<experiment_id> \\
        --formats markdown latex html \\
        --figures \\
        --tables \\
        --archive

Description
-----------
Loads an ExperimentSummary from a completed experiment output directory,
then orchestrates:

1. Figure generation (PNG, SVG, PDF)
2. Table export (CSV, JSON, Markdown, LaTeX)
3. Report generation (Markdown, LaTeX, HTML)
4. Archive assembly (manifest.json, CITATION.cff, CHECKLIST.md, README.md)

The script discovers experiment data by loading JSON files from the standard
ResultManager output structure.  If no matching files are found, it falls back
to generating a demo report using synthetic data.

Examples
--------
# Generate all formats for a completed experiment:
python scripts/generate_report.py --experiment-dir results/exp-001

# Generate only Markdown and HTML reports:
python scripts/generate_report.py --experiment-dir results/exp-001 \\
    --formats markdown html

# Run a demo with synthetic data (no experiment directory required):
python scripts/generate_report.py --demo --output-dir results/demo
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_reliability.reporting.summary import ExperimentSummary

# Ensure src/ is on the path when running from the repo root
_REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("generate_report")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(
        prog="generate_report.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--experiment-dir",
        type=pathlib.Path,
        default=None,
        help="Path to the experiment output directory produced by ExperimentRunner.",
    )
    p.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Root directory for generated outputs (defaults to --experiment-dir).",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        choices=["markdown", "latex", "html"],
        default=["markdown", "latex", "html"],
        help="Report formats to generate.",
    )
    p.add_argument(
        "--figures",
        action="store_true",
        default=True,
        help="Generate figures (default: True).",
    )
    p.add_argument(
        "--no-figures",
        action="store_false",
        dest="figures",
        help="Skip figure generation.",
    )
    p.add_argument(
        "--tables",
        action="store_true",
        default=True,
        help="Export tables (default: True).",
    )
    p.add_argument(
        "--archive",
        action="store_true",
        default=True,
        help="Assemble full archive (default: True).",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Generate a demo report using synthetic data.",
    )
    p.add_argument(
        "--skip-excel",
        action="store_true",
        default=False,
        help="Skip Excel table export (avoids openpyxl requirement).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    return p


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_summary_from_dir(exp_dir: pathlib.Path) -> ExperimentSummary:
    """Load an ExperimentSummary from a ResultManager output directory.

    Parameters
    ----------
    exp_dir : pathlib.Path
        Directory containing ``metrics.json``, ``rankings.json``, etc.

    Returns
    -------
    ExperimentSummary
    """
    from llm_reliability.records.evaluation import EvaluationRecord
    from llm_reliability.records.execution import ExecutionRecord
    from llm_reliability.records.metric import MetricRecord
    from llm_reliability.records.ranking import RankingRecord
    from llm_reliability.reporting.summary import ExperimentSummary

    def _load_json(fname: str) -> list:
        path = exp_dir / fname
        if not path.exists():
            logger.warning("File not found: %s", path)
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else []

    metrics_raw = _load_json("metrics.json")
    rankings_raw = _load_json("rankings.json")
    evals_raw = _load_json("evaluations.json")
    execs_raw = _load_json("executions.json")

    metrics = []
    for item in metrics_raw:
        try:
            metrics.append(MetricRecord.model_validate(item))
        except Exception as exc:
            logger.warning("MetricRecord parse failed: %s", exc)

    rankings = []
    for item in rankings_raw:
        try:
            rankings.append(RankingRecord.model_validate(item))
        except Exception as exc:
            logger.warning("RankingRecord parse failed: %s", exc)

    evaluations = []
    for item in evals_raw:
        try:
            evaluations.append(EvaluationRecord.model_validate(item))
        except Exception as exc:
            logger.warning("EvaluationRecord parse failed: %s", exc)

    executions = []
    for item in execs_raw:
        try:
            executions.append(ExecutionRecord.model_validate(item))
        except Exception as exc:
            logger.warning("ExecutionRecord parse failed: %s", exc)

    # Load experiment config for metadata
    config_path = exp_dir / "config.json"
    config_snapshot: dict = {}
    exp_name = exp_dir.name
    exp_id = exp_dir.name
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config_snapshot = cfg
            exp_name = cfg.get("experiment_name", exp_name)
            exp_id = cfg.get("experiment_id", exp_id)
        except Exception:
            pass

    return ExperimentSummary(
        experiment_id=exp_id,
        experiment_name=exp_name,
        metrics=metrics,
        rankings=rankings,
        evaluations=evaluations,
        executions=executions,
        config_snapshot=config_snapshot,
    )


def build_demo_summary() -> ExperimentSummary:
    """Build a synthetic ExperimentSummary for demo purposes."""
    from datetime import datetime, timezone

    from llm_reliability.records.metric import MetricRecord
    from llm_reliability.records.ranking import RankingRecord
    from llm_reliability.reporting.summary import ExperimentSummary

    ts = datetime.now(timezone.utc).isoformat()

    agents = [
        ("AgentAlpha", 0.85, 0.72, None, None),
        ("AgentBeta", 0.78, 0.81, 0.65, 0.70),
        ("AgentGamma", 0.92, 0.55, 0.48, None),
        ("AgentDelta", 0.61, 0.88, 0.79, 0.84),
    ]

    metrics = []
    for name, sr, cons, pert, ft in agents:
        comps = [cons]
        if pert is not None:
            comps.append(pert)
        if ft is not None:
            comps.append(ft)
        composite = sum(comps) / len(comps)
        m = MetricRecord(
            benchmark="MockBenchmark",
            agent=name,
            evaluation_count=20,
            success_rate=sr,
            repeated_run_consistency=cons,
            perturbation_robustness=pert,
            fault_tolerance=ft,
            composite_reliability=composite,
            computed_at=ts,
        )
        metrics.append(m)

    s_ranking = RankingRecord.from_metrics(metrics, ranking_type="success", computed_at=ts)
    r_ranking = RankingRecord.from_metrics(metrics, ranking_type="reliability", computed_at=ts)

    return ExperimentSummary(
        experiment_id="demo-exp-001",
        experiment_name="Demo: Success vs. Reliability Ranking Divergence",
        metrics=metrics,
        rankings=[s_ranking, r_ranking],
        config_snapshot={"mode": "demo", "benchmark": "MockBenchmark", "agents": 4},
        metadata={"note": "Synthetic demo data — not from a real experiment."},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load or build summary
    if args.demo:
        logger.info("Running in demo mode with synthetic data.")
        summary = build_demo_summary()
        output_dir = args.output_dir or pathlib.Path("results") / "demo"
    elif args.experiment_dir:
        if not args.experiment_dir.exists():
            logger.error("Experiment directory not found: %s", args.experiment_dir)
            return 1
        logger.info("Loading experiment from %s", args.experiment_dir)
        summary = load_summary_from_dir(args.experiment_dir)
        output_dir = args.output_dir or args.experiment_dir
    else:
        logger.error("Provide --experiment-dir or --demo.")
        parser.print_help()
        return 1

    logger.info(
        "Experiment: %s (%d agents, %d benchmarks, %d metrics, %d rankings)",
        summary.experiment_name,
        len(summary.agents),
        len(summary.benchmarks),
        len(summary.metrics),
        len(summary.rankings),
    )

    # Build archive
    if args.archive:
        try:
            from llm_reliability.reproducibility.archive import ArchiveBuilder

            builder = ArchiveBuilder()
            archive_dir = builder.build(
                summary,
                root_dir=output_dir,
                skip_excel=args.skip_excel,
                formats=args.formats,
            )
            logger.info("Archive assembled at: %s", archive_dir)
        except Exception as exc:
            logger.error("Archive build failed: %s", exc, exc_info=True)
            return 2

    else:
        # Individual steps
        exp_dir = output_dir / summary.experiment_id
        figures_dir = exp_dir / "figures"
        tables_dir = exp_dir / "tables"
        reports_dir = exp_dir / "reports"

        if args.figures:
            try:
                import matplotlib

                matplotlib.use("Agg")
                from llm_reliability.reproducibility.archive import \
                    ArchiveBuilder

                ArchiveBuilder()._generate_figures(summary, figures_dir)
                logger.info("Figures written to %s", figures_dir)
            except Exception as exc:
                logger.error("Figure generation failed: %s", exc)

        if args.tables:
            try:
                from llm_reliability.reproducibility.archive import \
                    ArchiveBuilder

                ArchiveBuilder()._generate_tables(summary, tables_dir, skip_excel=args.skip_excel)
                logger.info("Tables written to %s", tables_dir)
            except Exception as exc:
                logger.error("Table generation failed: %s", exc)

        try:
            from llm_reliability.reporting.report_generator import \
                ReportGenerator

            gen = ReportGenerator()
            paths = gen.generate(
                summary,
                output_dir=reports_dir,
                formats=args.formats,
                figure_dir=pathlib.Path("../figures"),
            )
            for fmt, path in paths.items():
                logger.info("Report [%s]: %s", fmt, path)
        except Exception as exc:
            logger.error("Report generation failed: %s", exc)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
