"""Global logging configuration and logger factory.

Call ``configure_logging()`` once at process startup to set up console and/or
file handlers.  Subsequent calls are no-ops; use ``force=True`` to replace
all existing handlers.
"""

from __future__ import annotations

import logging
import logging.handlers
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from llm_reliability.logging.formatters import ContextFilter, JsonFormatter

_configured = False


@dataclass
class LogConfig:
    """Configuration for the structured logging subsystem.

    Parameters
    ----------
    level : int | str
        Global log level.  Defaults to ``logging.INFO``.
    console : bool
        Whether to add a console handler.  Defaults to ``True``.
    console_level : int | str | None
        Console handler level; falls back to *level* when ``None``.
    file : str | Path | None
        Optional path to a log file.
    file_level : int | str | None
        File handler level; falls back to *level* when ``None``.
    file_max_bytes : int
        Max bytes per log file before rotation (default 10 MB).
    file_backup_count : int
        Number of rotated backup files to keep (default 5).
    format : Literal["plain", "json"]
        Output format.  ``"plain"`` uses a human-readable text format;
        ``"json"`` produces newline-delimited JSON.
    """

    level: int | str = logging.INFO
    console: bool = True
    console_level: int | str | None = None
    file: str | Path | None = None
    file_level: int | str | None = None
    file_max_bytes: int = 10 * 1024 * 1024
    file_backup_count: int = 5
    format: Literal["plain", "json"] = "plain"


def configure_logging(config: LogConfig | None = None, force: bool = False) -> None:
    """Configure the root logger with console and/or file handlers.

    Parameters
    ----------
    config : LogConfig | None
        Logging configuration.  If ``None``, a default ``LogConfig`` is used.
    force : bool
        If ``True``, remove all existing root handlers before applying the
        new configuration.  This allows tests to reconfigure logging.
    """
    global _configured
    if _configured and not force:
        return

    cfg = config or LogConfig()
    root = logging.getLogger()

    if force:
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()
        _configured = False

    if isinstance(cfg.level, str):
        cfg.level = getattr(logging, cfg.level.upper(), logging.INFO)
    root.setLevel(cfg.level)

    if cfg.format == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    if cfg.console:
        ch = logging.StreamHandler()
        ch_level = cfg.console_level if cfg.console_level is not None else cfg.level
        if isinstance(ch_level, str):
            ch_level = getattr(logging, ch_level.upper(), cfg.level)
        ch.setLevel(ch_level)
        ch.setFormatter(formatter)
        ch.addFilter(ContextFilter())
        root.addHandler(ch)

    if cfg.file:
        log_path = Path(cfg.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            str(log_path),
            maxBytes=cfg.file_max_bytes,
            backupCount=cfg.file_backup_count,
            encoding="utf-8",
        )
        fh_level = cfg.file_level if cfg.file_level is not None else cfg.level
        if isinstance(fh_level, str):
            fh_level = getattr(logging, fh_level.upper(), cfg.level)
        fh.setLevel(fh_level)
        fh.setFormatter(formatter)
        fh.addFilter(ContextFilter())
        root.addHandler(fh)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with the ``ContextFilter`` attached.

    Usage::

        logger = get_logger(__name__)
        logger.info("Benchmark starting")
    """
    logger = logging.getLogger(name)
    if not any(isinstance(f, ContextFilter) for f in logger.filters):
        logger.addFilter(ContextFilter())
    return logger
