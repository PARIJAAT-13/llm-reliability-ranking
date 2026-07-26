"""Tests for reproducibility checklist generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from llm_reliability.reproducibility.checklist import (
    CheckItem,
    ChecklistResult,
    ReproducibilityChecklist,
)


class TestCheckItem:
    def test_defaults(self) -> None:
        item = CheckItem(name="test", passed=True, message="ok")
        assert item.name == "test"
        assert item.passed
        assert not item.critical

    def test_critical_flag(self) -> None:
        item = CheckItem(name="critical", passed=True, message="ok", critical=True)
        assert item.critical


class TestChecklistResult:
    def test_all_passed_true(self) -> None:
        result = ChecklistResult(items=[CheckItem(name="a", passed=True, message="ok")])
        assert result.all_passed

    def test_all_passed_false(self) -> None:
        result = ChecklistResult(items=[CheckItem(name="a", passed=False, message="fail")])
        assert not result.all_passed

    def test_critical_passed(self) -> None:
        result = ChecklistResult(
            items=[
                CheckItem(name="a", passed=True, message="ok", critical=True),
                CheckItem(name="b", passed=False, message="fail", critical=False),
            ]
        )
        assert result.critical_passed
        assert not result.all_passed

    def test_n_passed(self) -> None:
        result = ChecklistResult(
            items=[
                CheckItem(name="a", passed=True, message="ok"),
                CheckItem(name="b", passed=False, message="fail"),
                CheckItem(name="c", passed=True, message="ok"),
            ]
        )
        assert result.n_passed == 2
        assert result.n_total == 3

    def test_markdown_contains_check_icons(self) -> None:
        result = ChecklistResult(
            items=[
                CheckItem(name="Check A", passed=True, message="ok"),
                CheckItem(name="Check B", passed=False, message="failed"),
            ]
        )
        md = result.markdown
        assert "✅" in md
        assert "❌" in md
        assert "Check A" in md
        assert "Check B" in md
        assert "2/3" not in md

    def test_markdown_summary_line(self) -> None:
        result = ChecklistResult(items=[CheckItem(name="A", passed=True, message="ok")])
        md = result.markdown
        assert "1/1" in md


@dataclass
class MockSummary:
    experiment_id: str = "exp-123"
    evaluations: list[Any] | None = None
    config_snapshot: dict[str, Any] | None = None
    metrics: list[Any] | None = None
    rankings: list[Any] | None = None
    executions: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.evaluations is None:
            self.evaluations = [{"seed": 42}]
        if self.config_snapshot is None:
            self.config_snapshot = {"base_seed": 42}
        if self.metrics is None:
            self.metrics = [{"agent": "A", "score": 0.9}]
        if self.rankings is None:
            self.rankings = [{"agent": "A", "rank": 1}]
        if self.executions is None:
            self.executions = [{"run_id": 1}]


class TestReproducibilityChecklist:
    def test_run_without_archive(self) -> None:
        checker = ReproducibilityChecklist()
        summary = MockSummary()
        result = checker.run(summary)
        assert result.n_total == 7

    def test_run_with_archive_checks_files(self, tmp_path: Path) -> None:
        checker = ReproducibilityChecklist()
        summary = MockSummary()
        result = checker.run(summary, archive_dir=str(tmp_path))
        assert result.n_total == 13

    def test_run_with_archive_all_files_present(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
        (tmp_path / "environment.json").write_text("{}", encoding="utf-8")
        (tmp_path / "CITATION.cff").write_text("", encoding="utf-8")
        (tmp_path / "README.md").write_text("", encoding="utf-8")
        (tmp_path / "figures").mkdir()
        (tmp_path / "figures" / "plot.png").write_text("", encoding="utf-8")
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / "report.md").write_text("", encoding="utf-8")

        checker = ReproducibilityChecklist()
        summary = MockSummary()
        result = checker.run(summary, archive_dir=str(tmp_path))
        assert result.all_passed

    def test_missing_experiment_id_fails(self) -> None:
        summary = MockSummary(experiment_id="")
        checker = ReproducibilityChecklist()
        result = checker.run(summary)
        assert not result.all_passed
        assert not result.critical_passed

    def test_save_creates_file(self, tmp_path: Path) -> None:
        result = ChecklistResult(items=[CheckItem(name="A", passed=True, message="ok")])
        checker = ReproducibilityChecklist()
        dest = checker.save(result, str(tmp_path / "CHECKLIST.md"))
        assert dest.exists()
        content = dest.read_text(encoding="utf-8")
        assert "Reproducibility Checklist" in content
