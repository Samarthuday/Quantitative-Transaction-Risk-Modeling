"""Structured logging configuration for the project."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure structured logging for a module.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default INFO)
        log_file: Optional path to write logs to file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ProgressLogger:
    """Context manager for tracking progress through large operations."""

    def __init__(self, logger: logging.Logger, total: int, message: str):
        self.logger = logger
        self.total = total
        self.message = message
        self.processed = 0

    def update(self, count: int = 1, interval: int = 10000):
        """Update progress and log at specified interval."""
        self.processed += count
        if self.processed % interval == 0:
            pct = (self.processed / self.total) * 100
            self.logger.info(
                f"{self.message}: {self.processed:,} / {self.total:,} ({pct:.1f}%)"
            )

    def __enter__(self):
        self.logger.info(f"Starting: {self.message} ({self.total:,} items)")
        return self

    def __exit__(self, *args):
        self.logger.info(f"Completed: {self.message}")
