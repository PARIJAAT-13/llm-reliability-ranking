"""Tests for the CLI interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from llm_reliability.cli import main


def test_cli_version(capsys):
    main(["--version"])
    captured = capsys.readouterr()
    assert "llm-reliability-ranking v" in captured.out


def test_cli_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])


def test_cli_list_benchmarks(capsys):
    main(["list", "benchmarks"])
    captured = capsys.readouterr()
    assert "Registered benchmarks" in captured.out


def test_cli_list_runtimes(capsys):
    main(["list", "runtimes"])
    captured = capsys.readouterr()
    assert "Registered runtimes" in captured.out


def test_cli_validate_config(tmp_path: Path, capsys):
    cfg = {
        "experiment_name": "cli_test",
        "benchmarks": [{"name": "mock", "dataset_path": "dummy.json"}],
        "agents": [{"name": "mock"}],
        "seeds": [0],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    main(["validate", str(config_path)])
    captured = capsys.readouterr()
    assert "Config valid" in captured.out


def test_cli_validate_config_missing(tmp_path: Path, capsys):
    with pytest.raises(SystemExit):
        main(["validate", str(tmp_path / "nonexistent.json")])
    captured = capsys.readouterr()
    assert "Error" in captured.err


def test_cli_validate_invalid_config(tmp_path: Path, capsys):
    config_path = tmp_path / "bad.json"
    config_path.write_text('{"invalid": true}', encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["validate", str(config_path)])
    captured = capsys.readouterr()
    assert "Config invalid" in captured.err


def test_cli_clear_cache(capsys):
    main(["clear-cache"])
    captured = capsys.readouterr()
    assert "Cache cleared" in captured.out
