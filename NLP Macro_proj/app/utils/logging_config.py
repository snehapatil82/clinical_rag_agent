"""
Centralized logging configuration for Medical RAG.
Provides clear, formatted, and non-sensitive logging across all modules.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "medical_rag",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configures and returns a structured logger.

    Args:
        name: Name of the logger.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Optional path to output log file.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    # Console handler with formatting
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Convenience function to get a child logger for a specific module."""
    return logging.getLogger(f"medical_rag.{module_name}")
