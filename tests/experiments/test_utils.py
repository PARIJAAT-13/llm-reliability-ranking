"""Tests for experiment utility functions (setup_experiment_logger)."""

import logging
from pathlib import Path

from llm_reliability.experiments.utils import setup_experiment_logger


class TestSetupExperimentLogger:
    def test_creates_log_directory(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs" / "subdir"
        logger = setup_experiment_logger(log_dir, "test-dir")
        assert log_dir.exists()
        assert logger.name == "experiment.test-dir"

    def test_returns_logger_instance(self, tmp_path: Path) -> None:
        logger = setup_experiment_logger(tmp_path, "test-instance")
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.DEBUG

    def test_has_console_and_file_handlers(self, tmp_path: Path) -> None:
        logger = setup_experiment_logger(tmp_path, "test-handlers")
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "StreamHandler" in handler_types
        assert "RotatingFileHandler" in handler_types

    def test_reuses_existing_handlers(self, tmp_path: Path) -> None:
        logger1 = setup_experiment_logger(tmp_path, "test-reuse")
        n_handlers = len(logger1.handlers)
        logger2 = setup_experiment_logger(tmp_path, "test-reuse")
        assert len(logger2.handlers) == n_handlers

    def test_creates_log_file(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        logger = setup_experiment_logger(log_dir, "test-file")
        logger.info("creating log file")
        del logger  # force flush
        log_file = log_dir / "test-file.log"
        assert log_file.exists()

    def test_different_experiments_different_loggers(self, tmp_path: Path) -> None:
        logger1 = setup_experiment_logger(tmp_path, "diff-1")
        logger2 = setup_experiment_logger(tmp_path, "diff-2")
        assert logger1.name != logger2.name

    def test_log_file_writable(self, tmp_path: Path) -> None:
        logger = setup_experiment_logger(tmp_path, "test-write")
        logger.info("test message")
        log_file = tmp_path / "test-write.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "test message" in content
