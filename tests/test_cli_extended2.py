"""Extended tests for the CLI — validate, export, discover, and utility commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_reliability.cli import main


def test_discover_models(capsys):
    main(["discover-models"])
    captured = capsys.readouterr()
    assert "runtime-specific" in captured.out


def test_discover_runtimes(capsys):
    main(["discover-runtimes"])
    captured = capsys.readouterr()
    assert "Registered runtimes" in captured.out


def test_list_benchmarks(capsys):
    main(["list", "benchmarks"])
    captured = capsys.readouterr()
    assert "Registered benchmarks" in captured.out or "No benchmarks" in captured.out


def test_list_runtimes(capsys):
    main(["list", "runtimes"])
    captured = capsys.readouterr()
    assert "Registered runtimes" in captured.out or "No runtimes" in captured.out


def test_hardware_info(capsys):
    main(["hardware-info"])
    captured = capsys.readouterr()
    assert "Hardware Profile" in captured.out


def test_system_info(capsys):
    main(["system-info"])
    captured = capsys.readouterr()
    assert "System Information" in captured.out


class TestValidateConfig:
    def test_validate_valid_config(self, tmp_path: Path, capsys):
        config = {
            "experiment_name": "test_validate",
            "benchmarks": [{"name": "mock", "dataset_path": "test.json"}],
            "agents": [{"name": "mock"}],
            "seeds": [42],
            "repetitions": 1,
            "llm": "mock",
            "prompt_version": "1",
            "dataset_version": "1",
        }
        path = tmp_path / "valid_config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        main(["validate", str(path)])
        captured = capsys.readouterr()
        assert "Config valid" in captured.out

    def test_validate_missing_file(self, tmp_path: Path, capsys):
        missing = tmp_path / "nonexistent.json"
        with pytest.raises(SystemExit):
            main(["validate", str(missing)])
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_validate_invalid_json(self, tmp_path: Path, capsys):
        path = tmp_path / "bad.json"
        path.write_text("{invalid json content", encoding="utf-8")
        with pytest.raises(SystemExit):
            main(["validate", str(path)])
        captured = capsys.readouterr()
        assert "Config invalid" in captured.err or "Error" in captured.err

    def test_validate_empty_benchmarks(self, tmp_path: Path, capsys):
        config = {
            "experiment_name": "no_benchmarks",
            "benchmarks": [],
            "agents": [{"name": "mock"}],
            "seeds": [42],
        }
        path = tmp_path / "no_bench.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with pytest.raises(SystemExit):
            main(["validate", str(path)])

    def test_validate_empty_agents(self, tmp_path: Path, capsys):
        config = {
            "experiment_name": "no_agents",
            "benchmarks": [{"name": "mock", "dataset_path": "test.json"}],
            "agents": [],
            "seeds": [42],
        }
        path = tmp_path / "no_agent.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with pytest.raises(SystemExit):
            main(["validate", str(path)])

    def test_validate_negative_seed(self, tmp_path: Path, capsys):
        config = {
            "experiment_name": "neg_seed",
            "benchmarks": [{"name": "mock", "dataset_path": "test.json"}],
            "agents": [{"name": "mock"}],
            "seeds": [-1],
        }
        path = tmp_path / "neg_seed.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with pytest.raises(SystemExit):
            main(["validate", str(path)])


class TestExportCommand:
    def test_export_no_directory(self, capsys):
        with pytest.raises(SystemExit):
            main(["export", "/nonexistent/export"])
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_export_empty_directory(self, tmp_path: Path, capsys):
        main(["export", str(tmp_path)])
        captured = capsys.readouterr()
        assert "No rankings found" in captured.out

    def test_export_csv_format_no_data(self, tmp_path: Path, capsys):
        main(["export", str(tmp_path), "--format", "csv"])
        captured = capsys.readouterr()
        assert "No rankings found" in captured.out

    def test_export_md_format_no_data(self, tmp_path: Path, capsys):
        main(["export", str(tmp_path), "--format", "md"])
        captured = capsys.readouterr()
        assert "No rankings found" in captured.out

    def test_export_latex_format_no_data(self, tmp_path: Path, capsys):
        main(["export", str(tmp_path), "--format", "latex"])
        captured = capsys.readouterr()
        assert "No rankings found" in captured.out

    def test_export_with_rankings(self, tmp_path: Path, capsys):
        rankings = [
            {
                "ranking_type": "success",
                "benchmark": "mock",
                "rankings": [["agent_a", 0.9]],
                "rank_map": {"agent_a": 1},
                "computed_at": "2026-01-01T00:00:00+00:00",
            }
        ]
        (tmp_path / "rankings.json").write_text(json.dumps(rankings), encoding="utf-8")
        main(["export", str(tmp_path), "--format", "csv"])
        captured = capsys.readouterr()
        assert "CSV:" in captured.out

    def test_export_all_formats(self, tmp_path: Path, capsys):
        rankings = [
            {
                "ranking_type": "success",
                "benchmark": "mock",
                "rankings": [["agent_a", 0.9]],
                "rank_map": {"agent_a": 1},
                "computed_at": "2026-01-01T00:00:00+00:00",
            }
        ]
        (tmp_path / "rankings.json").write_text(json.dumps(rankings), encoding="utf-8")
        main(["export", str(tmp_path), "--format", "all"])
        captured = capsys.readouterr()
        assert "CSV:" in captured.out
        assert "LaTeX:" in captured.out
        assert "Markdown:" in captured.out


class TestStatisticsCommand:
    def test_statistics_no_directory(self, capsys):
        with pytest.raises(SystemExit):
            main(["statistics", "/nonexistent/stats"])
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_statistics_empty_directory(self, tmp_path: Path, capsys):
        main(["statistics", str(tmp_path)])
        captured = capsys.readouterr()
        assert "No metrics found" in captured.out

    def test_statistics_with_metrics(self, tmp_path: Path, capsys):
        metrics = [
            {
                "benchmark": "mock",
                "agent": "agent_a",
                "evaluation_count": 5,
                "success_rate": 0.8,
                "repeated_run_consistency": 0.8,
                "composite_reliability": 0.8,
                "computed_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "benchmark": "mock",
                "agent": "agent_b",
                "evaluation_count": 5,
                "success_rate": 0.6,
                "repeated_run_consistency": 0.6,
                "composite_reliability": 0.6,
                "computed_at": "2026-01-01T00:00:00+00:00",
            },
        ]
        (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        main(["statistics", str(tmp_path)])
        captured = capsys.readouterr()
        assert "{" in captured.out or "0.8" in captured.out


class TestClearCache:
    def test_clear_cache(self, capsys):
        main(["clear-cache"])
        captured = capsys.readouterr()
        assert "Cache cleared" in captured.out


class TestHelp:
    def test_help_shows_commands(self, capsys):
        with pytest.raises(SystemExit):
            main(["--help"])
        captured = capsys.readouterr()
        assert "usage:" in captured.out

    def test_no_command_shows_help(self, capsys):
        main([])
        captured = capsys.readouterr()
        assert "usage:" in captured.out


def test_version(capsys):
    main(["--version"])
    captured = capsys.readouterr()
    assert "llm-reliability-ranking" in captured.out
