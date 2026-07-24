"""Tests for structured logging subsystem."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from llm_reliability.logging import LogConfig, LogContext, configure_logging, get_logger
from llm_reliability.logging.formatters import ContextFilter, JsonFormatter


def _close_root_handlers() -> None:
    """Close and remove all handlers from the root logger."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()


@pytest.fixture(autouse=True)
def _reset_logging():
    """Remove and close all root handlers, reset config before each test."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    root.setLevel(logging.WARNING)
    import llm_reliability.logging.config as lcfg
    lcfg._configured = False
    yield


# ---------------------------------------------------------------------------
# Logger creation
# ---------------------------------------------------------------------------


def test_get_logger_returns_logger():
    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.module"


def test_get_logger_has_context_filter():
    logger = get_logger("test.filter")
    assert any(isinstance(f, ContextFilter) for f in logger.filters)


def test_get_logger_reuses_existing():
    logger1 = get_logger("test.reuse")
    logger2 = get_logger("test.reuse")
    assert logger1 is logger2


# ---------------------------------------------------------------------------
# Configurable log level
# ---------------------------------------------------------------------------


def test_configure_logging_sets_level():
    cfg = LogConfig(level=logging.WARNING, console=False)
    configure_logging(cfg, force=True)
    root = logging.getLogger()
    assert root.level <= logging.WARNING
    _close_root_handlers()


def test_configure_logging_string_level():
    cfg = LogConfig(level="ERROR", console=False)
    configure_logging(cfg, force=True)
    root = logging.getLogger()
    assert root.level <= logging.ERROR
    _close_root_handlers()


# ---------------------------------------------------------------------------
# File logging
# ---------------------------------------------------------------------------


def test_file_logging_writes_output(tmp_path: Path):
    log_file = tmp_path / "test.log"
    cfg = LogConfig(level=logging.DEBUG, console=False, file=str(log_file))
    configure_logging(cfg, force=True)
    test_logger = get_logger("test.file")
    test_logger.info("Hello from file test")
    _close_root_handlers()
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Hello from file test" in content


def test_file_logging_respects_level(tmp_path: Path):
    log_file = tmp_path / "quiet.log"
    cfg = LogConfig(level=logging.WARNING, console=False, file=str(log_file))
    configure_logging(cfg, force=True)
    test_logger = get_logger("test.quiet")
    test_logger.info("Should not appear")
    test_logger.warning("Should appear")
    _close_root_handlers()
    content = log_file.read_text(encoding="utf-8")
    assert "Should not appear" not in content
    assert "Should appear" in content


# ---------------------------------------------------------------------------
# Structured log format (JSON)
# ---------------------------------------------------------------------------


def test_json_formatter_output():
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="test.json",
        level=logging.INFO,
        pathname=__file__,
        lineno=100,
        msg="JSON test message",
        args=(),
        exc_info=None,
    )
    output = fmt.format(record)
    parsed = json.loads(output)
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "JSON test message"
    assert parsed["logger"] == "test.json"
    assert "time" in parsed


def test_json_formatter_includes_extra():
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="test.extra",
        level=logging.INFO,
        pathname=__file__,
        lineno=100,
        msg="With extras",
        args=(),
        exc_info=None,
    )
    record.experiment_id = "exp-001"
    record.benchmark = "agentboard"
    output = fmt.format(record)
    parsed = json.loads(output)
    assert parsed["experiment_id"] == "exp-001"
    assert parsed["benchmark"] == "agentboard"


def test_json_formatter_with_log_context():
    fmt = JsonFormatter()
    with LogContext(experiment_id="ctx-001", run_index=5):
        record = logging.LogRecord(
            name="test.ctx",
            level=logging.INFO,
            pathname=__file__,
            lineno=100,
            msg="Context test",
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["experiment_id"] == "ctx-001"
        assert parsed["run_index"] == 5


def test_configure_logging_json_format(tmp_path: Path):
    log_file = tmp_path / "structured.log"
    cfg = LogConfig(level=logging.INFO, console=False, file=str(log_file), format="json")
    configure_logging(cfg, force=True)
    test_logger = get_logger("test.structured")
    test_logger.info("Structured message")
    _close_root_handlers()
    content = log_file.read_text(encoding="utf-8").strip()
    parsed = json.loads(content)
    assert parsed["message"] == "Structured message"
    assert parsed["logger"] == "test.structured"


# ---------------------------------------------------------------------------
# LogContext
# ---------------------------------------------------------------------------


def test_log_context_adds_fields():
    with LogContext(experiment_id="exp-999"):
        ctx = __import__("llm_reliability.logging.context", fromlist=["get_log_context"]).get_log_context()
        assert ctx["experiment_id"] == "exp-999"


def test_log_context_restores_after_exit():
    with LogContext(experiment_id="exp-999"):
        pass
    ctx = __import__("llm_reliability.logging.context", fromlist=["get_log_context"]).get_log_context()
    assert "experiment_id" not in ctx


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_standard_logging_still_works():
    std_logger = logging.getLogger("test.std_compat")
    std_logger.info("Standard log message still works")


def test_configure_logging_called_twice_is_noop():
    cfg = LogConfig(level=logging.INFO)
    configure_logging(cfg)
    configure_logging(cfg)  # Should not raise


# ---------------------------------------------------------------------------
# Integration: experiment logging via ExperimentRunner
# ---------------------------------------------------------------------------


def test_experiment_runner_creates_logger():
    from llm_reliability.experiments.experiment_runner import ExperimentRunner
    from llm_reliability.experiments.experiment_models import ExperimentSpec, BenchmarkSpec, AgentSpec

    spec = ExperimentSpec(
        experiment_name="log_test",
        benchmarks=[BenchmarkSpec(name="mock", dataset_path="dummy.json")],
        agents=[AgentSpec(name="mock")],
        seeds=[0],
    )
    runner = ExperimentRunner(spec, agent_factory=lambda a, c: None)
    assert runner._logger is not None


# ---------------------------------------------------------------------------
# CLI / demo helpers
# ---------------------------------------------------------------------------


def test_log_events_format(tmp_path: Path):
    """Verify that structured log events at key lifecycle points contain expected fields."""
    log_file = tmp_path / "events.log"
    cfg = LogConfig(level=logging.DEBUG, console=False, file=str(log_file), format="json")
    configure_logging(cfg, force=True)

    logger = get_logger("test.events")

    with LogContext(experiment_id="demo-exp", benchmark="demo-bench"):
        logger.info("Experiment started", extra={"event": "experiment_start", "total_runs": 5})
        logger.info("Cache hit", extra={"event": "cache_hit", "cache_key": "abc123"})
        logger.info("Benchmark run completed", extra={
            "event": "benchmark_complete", "model": "gpt-4",
            "duration_seconds": 12.345, "num_executions": 10,
        })
        logger.error("Benchmark run failed", extra={
            "event": "benchmark_failure", "model": "gpt-4",
            "error": "Connection timeout",
        })

    _close_root_handlers()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4

    # Experiment start event
    ev1 = json.loads(lines[0])
    assert ev1["event"] == "experiment_start"
    assert ev1["total_runs"] == 5
    assert ev1["experiment_id"] == "demo-exp"

    # Cache hit event
    ev2 = json.loads(lines[1])
    assert ev2["event"] == "cache_hit"
    assert ev2["cache_key"] == "abc123"

    # Benchmark complete event
    ev3 = json.loads(lines[2])
    assert ev3["event"] == "benchmark_complete"
    assert ev3["model"] == "gpt-4"
    assert ev3["duration_seconds"] == 12.345

    # Failure event
    ev4 = json.loads(lines[3])
    assert ev4["event"] == "benchmark_failure"
    assert "Connection timeout" in ev4["error"]
