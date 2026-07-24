"""
Reproducibility checklist generator.

Purpose
-------
Run automated checks on an experiment to verify that it meets minimum
reproducibility standards and produce a Markdown checklist report.

Responsibilities
----------------
- Verify deterministic seeds are present
- Verify configuration snapshot is non-empty
- Verify all artifact types are present (executions, evaluations, metrics, rankings)
- Verify environment snapshot exists
- Verify manifest.json exists
- Produce a Markdown checklist with pass/fail indicators

Usage example
-------------
>>> from llm_reliability.reproducibility.checklist import ReproducibilityChecklist
>>> checker = ReproducibilityChecklist()
>>> result = checker.run(summary, archive_dir="results/exp-001")
>>> print(result.markdown)
>>> checker.save(result, "results/exp-001/CHECKLIST.md")

How the checklist is produced
------------------------------
``run()`` executes a list of ``CheckItem`` callables against the experiment
summary and archive directory.  Each item returns a pass/fail status and a
message.  The aggregated result is formatted as a GitHub-flavoured Markdown
checklist.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckItem:
    """A single reproducibility check."""

    name: str
    passed: bool
    message: str
    critical: bool = False


@dataclass
class ChecklistResult:
    """Result of running all reproducibility checks."""

    items: list[CheckItem] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """True if every check passed."""
        return all(item.passed for item in self.items)

    @property
    def critical_passed(self) -> bool:
        """True if every critical check passed."""
        return all(item.passed for item in self.items if item.critical)

    @property
    def n_passed(self) -> int:
        return sum(1 for item in self.items if item.passed)

    @property
    def n_total(self) -> int:
        return len(self.items)

    @property
    def markdown(self) -> str:
        """Render the checklist as Markdown."""
        lines = [
            "# Reproducibility Checklist\n",
            f"**Result**: {self.n_passed}/{self.n_total} checks passed  ",
            f"**Status**: {'✅ PASS' if self.all_passed else '⚠️ PARTIAL' if self.critical_passed else '❌ FAIL'}\n",
            "## Checks\n",
        ]
        for item in self.items:
            icon = "✅" if item.passed else "❌"
            crit = " *(critical)*" if item.critical else ""
            lines.append(f"- {icon} **{item.name}**{crit}: {item.message}")

        return "\n".join(lines)


class ReproducibilityChecklist:
    """Runs automated reproducibility checks on an experiment."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        summary: Any,
        archive_dir: str | pathlib.Path | None = None,
    ) -> ChecklistResult:
        """Execute all reproducibility checks.

        Parameters
        ----------
        summary : ExperimentSummary
        archive_dir : str | Path | None
            Root of the experiment output directory.  If provided, file-
            presence checks are also run.

        Returns
        -------
        ChecklistResult
        """
        result = ChecklistResult()
        archive_path = pathlib.Path(archive_dir) if archive_dir else None

        checks = [
            self._check_experiment_id(summary),
            self._check_seeds_present(summary),
            self._check_config_snapshot(summary),
            self._check_metrics_present(summary),
            self._check_rankings_present(summary),
            self._check_evaluations_present(summary),
            self._check_executions_present(summary),
        ]

        if archive_path:
            checks += [
                self._check_file_exists(archive_path, "manifest.json", critical=True),
                self._check_file_exists(archive_path, "environment.json"),
                self._check_file_exists(archive_path, "CITATION.cff"),
                self._check_file_exists(archive_path, "README.md"),
                self._check_figures_dir(archive_path),
                self._check_reports_dir(archive_path),
            ]

        result.items = checks
        return result

    def save(
        self,
        result: ChecklistResult,
        path: str | pathlib.Path,
    ) -> pathlib.Path:
        """Save the Markdown checklist to *path*.

        Parameters
        ----------
        result : ChecklistResult
        path : str | Path

        Returns
        -------
        pathlib.Path
        """
        dest = pathlib.Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result.markdown, encoding="utf-8")
        return dest

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_experiment_id(summary: Any) -> CheckItem:
        eid = getattr(summary, "experiment_id", "")
        ok = bool(eid and eid != "unknown")
        return CheckItem(
            name="Experiment ID",
            passed=ok,
            message=f"`{eid}`" if ok else "Missing or empty experiment_id.",
            critical=True,
        )

    @staticmethod
    def _check_seeds_present(summary: Any) -> CheckItem:
        evals = getattr(summary, "evaluations", [])
        seeds = set()
        for ev in evals:
            if isinstance(ev, dict):
                seed = ev.get("seed")
            else:
                seed = getattr(ev, "seed", None)
            if seed is not None:
                seeds.add(seed)
        ok = bool(seeds)
        return CheckItem(
            name="Deterministic Seeds",
            passed=ok,
            message=f"Seeds used: {sorted(seeds)}" if ok else "No seeds found in evaluations.",
            critical=True,
        )

    @staticmethod
    def _check_config_snapshot(summary: Any) -> CheckItem:
        snap = getattr(summary, "config_snapshot", {})
        ok = bool(snap)
        return CheckItem(
            name="Configuration Snapshot",
            passed=ok,
            message=f"{len(snap)} key(s) captured." if ok else "config_snapshot is empty.",
            critical=True,
        )

    @staticmethod
    def _check_metrics_present(summary: Any) -> CheckItem:
        n = len(getattr(summary, "metrics", []))
        ok = n > 0
        return CheckItem(
            name="Metric Records",
            passed=ok,
            message=f"{n} MetricRecord(s) present." if ok else "No MetricRecords found.",
        )

    @staticmethod
    def _check_rankings_present(summary: Any) -> CheckItem:
        n = len(getattr(summary, "rankings", []))
        ok = n > 0
        return CheckItem(
            name="Ranking Records",
            passed=ok,
            message=f"{n} RankingRecord(s) present." if ok else "No RankingRecords found.",
        )

    @staticmethod
    def _check_evaluations_present(summary: Any) -> CheckItem:
        n = len(getattr(summary, "evaluations", []))
        ok = n > 0
        return CheckItem(
            name="Evaluation Records",
            passed=ok,
            message=f"{n} EvaluationRecord(s) present." if ok else "No EvaluationRecords found.",
        )

    @staticmethod
    def _check_executions_present(summary: Any) -> CheckItem:
        n = len(getattr(summary, "executions", []))
        ok = n > 0
        return CheckItem(
            name="Execution Records",
            passed=ok,
            message=f"{n} ExecutionRecord(s) present." if ok else "No ExecutionRecords found.",
        )

    @staticmethod
    def _check_file_exists(
        base: pathlib.Path,
        filename: str,
        critical: bool = False,
    ) -> CheckItem:
        path = base / filename
        ok = path.exists()
        return CheckItem(
            name=f"File: {filename}",
            passed=ok,
            message=f"Found at `{path}`." if ok else f"Missing: `{path}`.",
            critical=critical,
        )

    @staticmethod
    def _check_figures_dir(base: pathlib.Path) -> CheckItem:
        d = base / "figures"
        ok = d.is_dir() and any(d.iterdir())
        return CheckItem(
            name="Figures Directory",
            passed=ok,
            message=(
                "Non-empty figures/ directory found." if ok else "figures/ is missing or empty."
            ),
        )

    @staticmethod
    def _check_reports_dir(base: pathlib.Path) -> CheckItem:
        d = base / "reports"
        ok = d.is_dir() and any(d.iterdir())
        return CheckItem(
            name="Reports Directory",
            passed=ok,
            message=(
                "Non-empty reports/ directory found." if ok else "reports/ is missing or empty."
            ),
        )
