"""Comprehensive tests for the structured logging module."""

from __future__ import annotations

import json
import logging
from io import StringIO

import pytest

from llm_reliability.logging import (
    JsonFormatter,
    LogConfig,
    LogContext,
    configure_logging,
    get_log_context,
    get_logger,
)
from llm_reliability.logging.formatters import ContextFilter

# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    @pytest.fixture(autouse=True)
    def _reset_context(self):
        get_log_context().clear()

    def make_record(
        self, msg="test message", level=logging.INFO, name="test_logger", exc_info=None, extra=None
    ) -> logging.LogRecord:
        record = logging.LogRecord(name, level, "test.py", 42, msg, (), exc_info)
        if extra:
            for k, v in extra.items():
                setattr(record, k, v)
        return record

    def test_json_formatter_basic(self):
        fmt = JsonFormatter()
        record = self.make_record()
        output = fmt.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_json_formatter_includes_timestamp(self):
        fmt = JsonFormatter()
        record = self.make_record()
        output = fmt.format(record)
        parsed = json.loads(output)
        assert "time" in parsed
        assert isinstance(parsed["time"], str)

    def test_json_formatter_includes_level(self):
        fmt = JsonFormatter()
        record = self.make_record(level=logging.WARNING)
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "WARNING"

    def test_json_formatter_includes_logger_name(self):
        fmt = JsonFormatter()
        record = self.make_record(name="my.custom.logger")
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["logger"] == "my.custom.logger"

    def test_json_formatter_includes_message(self):
        fmt = JsonFormatter()
        record = self.make_record(msg="hello world")
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello world"

    def test_json_formatter_includes_exception(self):
        fmt = JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
            record = self.make_record(exc_info=exc_info)
        output = fmt.format(record)
        parsed = json.loads(output)
        assert "exc_info" in parsed
        assert "boom" in parsed["exc_info"]

    def test_json_formatter_handles_extra_fields(self):
        fmt = JsonFormatter()
        record = self.make_record(extra={"experiment_id": "abc-123", "benchmark": "test"})
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["experiment_id"] == "abc-123"
        assert parsed["benchmark"] == "test"


# ---------------------------------------------------------------------------
# ContextFilter
# ---------------------------------------------------------------------------


class TestContextFilter:
    @pytest.fixture(autouse=True)
    def _reset_context(self):
        get_log_context().clear()

    def test_context_filter_filters_extra_fields(self):
        f = ContextFilter()
        with LogContext(experiment_id="ctx-1", run_id=42):
            record = logging.LogRecord("test", logging.INFO, "test.py", 42, "msg", (), None)
            assert f.filter(record)
            assert getattr(record, "experiment_id", None) == "ctx-1"
            assert getattr(record, "run_id", None) == 42


# ---------------------------------------------------------------------------
# LogContext
# ---------------------------------------------------------------------------


class TestLogContext:
    @pytest.fixture(autouse=True)
    def _reset_context(self):
        get_log_context().clear()

    def test_log_context_single(self):
        with LogContext(experiment_id="exp-123"):
            ctx = get_log_context()
            assert ctx["experiment_id"] == "exp-123"

    def test_log_context_nested(self):
        with LogContext(outer="a"):
            with LogContext(inner="b"):
                ctx = get_log_context()
                assert "outer" not in ctx
                assert ctx["inner"] == "b"
            ctx = get_log_context()
            assert ctx["outer"] == "a"
            assert "inner" not in ctx

    def test_log_context_no_interference(self):
        with LogContext(experiment_id="first"):
            pass
        with LogContext(experiment_id="second"):
            ctx = get_log_context()
            assert ctx["experiment_id"] == "second"

    def test_log_context_empty(self):
        with LogContext():
            ctx = get_log_context()
            assert ctx == {}

    def test_log_context_unicode(self):
        with LogContext(label="\u00e9\u00e0\u00fc\u00f1"):
            ctx = get_log_context()
            assert ctx["label"] == "\u00e9\u00e0\u00fc\u00f1"

    def test_log_context_many_fields(self):
        fields = {f"key_{i}": i for i in range(100)}
        with LogContext(**fields):
            ctx = get_log_context()
            assert len(ctx) == 100
            assert ctx["key_0"] == 0
            assert ctx["key_99"] == 99


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    @pytest.fixture(autouse=True)
    def _reset_configured_flag(self, monkeypatch: pytest.MonkeyPatch):
        import llm_reliability.logging.config as cfg_module

        monkeypatch.setattr(cfg_module, "_configured", False)
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()

    def test_configure_logging_basic(self):
        configure_logging()
        root = logging.getLogger()
        assert len(root.handlers) > 0

    def test_configure_logging_level(self):
        configure_logging(LogConfig(level=logging.DEBUG))
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_configure_logging_force(self):
        configure_logging(LogConfig(level=logging.INFO))
        configure_logging(LogConfig(level=logging.DEBUG), force=True)
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_configure_logging_json(self, capsys):
        configure_logging(LogConfig(format="json"))
        logger = logging.getLogger("json_test")
        logger.info("hello json")
        captured = capsys.readouterr()
        output = (captured.out or captured.err).strip()
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "hello json"
        assert parsed["logger"] == "json_test"
        assert "time" in parsed


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_logger_creation(self):
        logger = get_logger("my.module")
        assert logger.name == "my.module"
        assert isinstance(logger, logging.Logger)

    def test_logger_creation_has_context_filter(self):
        logger = get_logger("my.module")
        has_filter = any(isinstance(f, ContextFilter) for f in logger.filters)
        assert has_filter
