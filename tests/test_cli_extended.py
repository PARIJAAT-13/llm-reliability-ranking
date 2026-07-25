"""Tests for the enhanced CLI with all new commands."""

import json
import tempfile
from pathlib import Path

import pytest

from llm_reliability.cli import main


def test_version(capsys):
    main(["--version"])
    captured = capsys.readouterr()
    assert "llm-reliability-ranking" in captured.out


def test_no_command_shows_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert "usage:" in captured.out


def test_list_benchmarks(capsys):
    main(["list", "benchmarks"])
    captured = capsys.readouterr()
    assert "Registered benchmarks" in captured.out or "No benchmarks" in captured.out


def test_list_runtimes(capsys):
    main(["list", "runtimes"])
    captured = capsys.readouterr()
    assert "Registered runtimes" in captured.out or "No runtimes" in captured.out


def test_discover_runtimes(capsys):
    main(["discover-runtimes"])
    captured = capsys.readouterr()
    assert "Registered runtimes" in captured.out


def test_hardware_info(capsys):
    main(["hardware-info"])
    captured = capsys.readouterr()
    assert "Hardware Profile" in captured.out


def test_system_info(capsys):
    main(["system-info"])
    captured = capsys.readouterr()
    assert "System Information" in captured.out


def test_validate_invalid_path(capsys):
    with pytest.raises(SystemExit):
        main(["validate", "/nonexistent/path.json"])
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_validate_valid_config(capsys, tmp_path):
    config = {
        "experiment_name": "test",
        "benchmarks": [{"name": "mock", "dataset_path": "test.json"}],
        "agents": [{"name": "mock"}],
        "seeds": [42],
        "repetitions": 1,
        "llm": "mock",
        "prompt_version": "1",
        "dataset_version": "1",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    main(["validate", str(path)])
    captured = capsys.readouterr()
    assert "Config valid" in captured.out


def test_checkpoint_no_dir(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["checkpoint", "/nonexistent"])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_checkpoint_empty_dir(capsys, tmp_path):
    main(["checkpoint", str(tmp_path)])
    captured = capsys.readouterr()
    assert "No checkpoint found" in captured.out


def test_report_no_dir(capsys):
    with pytest.raises(SystemExit):
        main(["report", "/nonexistent"])
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_export_no_dir(capsys):
    with pytest.raises(SystemExit):
        main(["export", "/nonexistent"])
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_statistics_no_dir(capsys):
    with pytest.raises(SystemExit):
        main(["statistics", "/nonexistent"])
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_clear_cache(capsys):
    main(["clear-cache"])
    captured = capsys.readouterr()
    assert "Cache cleared" in captured.out


def test_discover_models(capsys):
    main(["discover-models"])
    captured = capsys.readouterr()
    assert "runtime-specific" in captured.out


def test_compare_multiple(tmp_path, capsys):
    d1 = tmp_path / "exp1"
    d2 = tmp_path / "exp2"
    d1.mkdir()
    d2.mkdir()
    main(["compare", str(d1), str(d2)])
    captured = capsys.readouterr()
    assert "No experiment summaries" in captured.out
