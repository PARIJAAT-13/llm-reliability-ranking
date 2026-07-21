"""
Utility functions for the Experiment Runner.

Provides logging setup and helper functions shared across the runner modules.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def setup_experiment_logger(log_dir: Path, experiment_id: str) -> logging.Logger:
    """Configure a named logger that writes to both console and a rotating file.

    Parameters
    ----------
    log_dir : Path
        Directory where the log file will be stored.
    experiment_id : str
        Unique identifier used to name the log file.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{experiment_id}.log"

    logger = logging.getLogger(f"experiment.{experiment_id}")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # Rotating file handler (10 MB, 5 backups)
        fh = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
