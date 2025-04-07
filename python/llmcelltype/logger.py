"""
Logging module for LLMCellType.
"""

import os
import datetime
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create logger
logger = logging.getLogger('llmcelltype')

# Default log directory
DEFAULT_LOG_DIR = os.path.join(os.path.expanduser('~'), '.llmcelltype', 'logs')

def setup_logging(log_dir: Optional[str] = None, log_level: str = 'INFO') -> None:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to store log files. If None, uses default directory.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Set log directory
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'llmcelltype_{timestamp}.log')
    
    # Set log level
    level = getattr(logging, log_level.upper())
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        '%Y-%m-%d %H:%M:%S'
    ))
    
    # Add handler to logger
    logger.addHandler(file_handler)
    logger.setLevel(level)
    
    logger.info(f"Logging initialized. Log file: {log_file}")

def write_log(message: str, level: str = 'INFO') -> None:
    """
    Write a message to the log.
    
    Args:
        message: Message to log
        level: Log level (debug, info, warning, error, critical)
    """
    level_method = getattr(logger, level.lower())
    level_method(message)

def get_logger():
    """
    Get the logger instance.
    
    Returns:
        Logger: The logger instance
    """
    return logger
