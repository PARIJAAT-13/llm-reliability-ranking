"""Extended CLI tests covering error paths, help output, and edge cases."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_reliability.cli import main

# ---------------------------------------------------------------------------
# run command — error paths
# ---------------------------------------------------------------------------


class TestCliRunErrors:
    def test_cli_run_nonexistent_config(self, capsys):
        with pytest.raises(SystemExit):
            main(["run", "/nonexistent/path/config.json"])
        captured = capsys.readouterr()
        assert "Error" in captured.err
        assert "not found" in captured.err

    def test_cli_run_bad_json(self, tmp_path: Path):
        config_path = tmp_path / "bad.json"
        config_path.write_text("not even json", encoding="utf-8")
        with pytest.raises(Exception):
            main(["run", str(config_path)])


# ---------------------------------------------------------------------------
# help output for subcommands
# ---------------------------------------------------------------------------


class TestCliSubcommandHelp:
    def test_cli_list_help(self, capsys):
        with pytest.raises(SystemExit):
            main(["list", "--help"])
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()
        assert "benchmarks" in captured.out
        assert "runtimes" in captured.out

    def test_cli_run_help(self, capsys):
        with pytest.raises(SystemExit):
            main(["run", "--help"])
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()
        assert "--no-cache" in captured.out

    def test_cli_validate_help(self, capsys):
        with pytest.raises(SystemExit):
            main(["validate", "--help"])
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()
        assert "config" in captured.out

    def test_cli_clear_cache_help(self, capsys):
        with pytest.raises(SystemExit):
            main(["clear-cache", "--help"])
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()


# ---------------------------------------------------------------------------
# invalid command
# ---------------------------------------------------------------------------


class TestCliInvalidCommand:
    def test_cli_invalid_command(self, capsys):
        with pytest.raises(SystemExit):
            main(["does-not-exist"])
        captured = capsys.readouterr()
        assert "error" in captured.err.lower() or "invalid" in captured.err.lower()


# ---------------------------------------------------------------------------
# list commands output
# ---------------------------------------------------------------------------


class TestCliListOutput:
    def test_cli_list_benchmarks_actual(self, capsys):
        main(["list", "benchmarks"])
        captured = capsys.readouterr()
        assert "Registered benchmarks" in captured.out
        assert "GSM8K" in captured.out or "MMLU" in captured.out or "ARC" in captured.out

    def test_cli_list_runtimes_actual(self, capsys):
        main(["list", "runtimes"])
        captured = capsys.readouterr()
        assert "Registered runtimes" in captured.out
        assert "mock" in captured.out or "gpt" in captured.out


# ---------------------------------------------------------------------------
# version output
# ---------------------------------------------------------------------------


class TestCliVersion:
    def test_cli_version_output_format(self, capsys):
        main(["--version"])
        captured = capsys.readouterr()
        assert captured.out.startswith("llm-reliability-ranking v")
        assert "." in captured.out


# ---------------------------------------------------------------------------
# validate with valid config
# ---------------------------------------------------------------------------


class TestCliValidate:
    def test_cli_validate_with_valid_config(self, tmp_path: Path, capsys):
        cfg = {
            "experiment_name": "test",
            "experiment_id": "test-1",
            "benchmarks": [{"name": "mock", "dataset_path": "dummy.json"}],
            "agents": [{"name": "mock"}],
            "seeds": [42],
            "repetitions": 1,
        }
        config_path = tmp_path / "valid_config.json"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
        main(["validate", str(config_path)])
        captured = capsys.readouterr()
        assert "Config valid" in captured.out
        assert "test-1" in captured.out


# ---------------------------------------------------------------------------
# run with --no-cache flag
# ---------------------------------------------------------------------------


class TestCliRun:
    def test_cli_run_with_no_cache_flag(self, tmp_path: Path, capsys):
        cfg = {
            "experiment_name": "test",
            "experiment_id": "test-1",
            "benchmarks": [{"name": "mock", "dataset_path": "dummy.json"}],
            "agents": [{"name": "mock"}],
            "seeds": [42],
            "repetitions": 1,
        }
        config_path = tmp_path / "run_config.json"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")

        mock_status = type(
            "MockStatus", (), {"state": "completed", "completed_runs": 1, "failed_runs": 0}
        )()

        with patch("llm_reliability.experiments.experiment_runner.ExperimentRunner") as MockRunner:
            instance = MockRunner.return_value
            instance.run.return_value = mock_status
            main(["run", str(config_path), "--no-cache"])
        captured = capsys.readouterr()
        assert "completed" in captured.out
