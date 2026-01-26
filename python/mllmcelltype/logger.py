"""Logging module for LLMCellType."""

from __future__ import annotations

import datetime
import logging
import os
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create logger
logger = logging.getLogger("llmcelltype")

# Default log directory
DEFAULT_LOG_DIR = os.path.join(os.path.expanduser("~"), ".mllmcelltype", "logs")

# Track initialization state for idempotent setup
_logging_initialized = False
_current_log_file: Optional[str] = None


def setup_logging(log_dir: Optional[str] = None, log_level: str = "INFO") -> None:
    """Setup logging configuration.

    This function is idempotent - multiple calls with the same parameters
    will not create duplicate handlers. If called with different parameters,
    old handlers are cleaned up before adding new ones.

    Args:
        log_dir: Directory to store log files. If None, uses default directory.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    """
    global _logging_initialized, _current_log_file

    # Set log directory
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    # Set log level
    level = getattr(logging, log_level.upper())

    # If already initialized, just update level and return
    if _logging_initialized:
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
        return

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"llmcelltype_{timestamp}.log")

    # Remove any existing file handlers to prevent duplication
    for handler in logger.handlers[:]:  # Use slice copy to safely remove during iteration
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )

    # Add handler to logger
    logger.addHandler(file_handler)
    logger.setLevel(level)

    # Mark as initialized
    _logging_initialized = True
    _current_log_file = log_file

    logger.info(f"Logging initialized. Log file: {log_file}")


def write_log(message: str, level: str = "INFO") -> None:
    """Write a message to the log.

    Args:
        message: Message to log
        level: Log level (debug, info, warning, error, critical)

    """
    level_method = getattr(logger, level.lower())
    level_method(message)


def reset_logging() -> None:
    """Reset logging state to allow re-initialization.

    This is useful for testing or when you need to change the log directory
    after initial setup. It closes all file handlers and resets the
    initialization flag.
    """
    global _logging_initialized, _current_log_file

    # Close and remove all file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    # Reset state
    _logging_initialized = False
    _current_log_file = None


def get_current_log_file() -> Optional[str]:
    """Get the path to the current log file.

    Returns:
        The path to the current log file, or None if logging is not initialized.
    """
    return _current_log_file
