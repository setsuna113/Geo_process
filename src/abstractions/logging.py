"""
Logging utilities - stop copy-pasting logger setup.

Used 50+ times across the codebase.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance. Used 50+ times.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_logging(
    level: Union[str, int] = logging.INFO,
    format: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True
) -> None:
    """Setup logging configuration. Used 5+ times in main scripts.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log message format string
        log_file: Optional file to write logs to
        console: Whether to also log to console
    """
    if format is None:
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    handlers = []
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(format))
        handlers.append(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format,
        handlers=handlers
    )


def create_experiment_logger(
    experiment_name: str,
    output_dir: Union[str, Path],
    level: Union[str, int] = logging.INFO
) -> logging.Logger:
    """Create logger for an experiment. Used 8+ times.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Directory for log files
        level: Logging level
        
    Returns:
        Configured logger for the experiment
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_dir / f"{experiment_name}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(f"experiment.{experiment_name}")
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler with simpler format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Experiment logger created: {experiment_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger


def log_dict(logger: logging.Logger, data: dict, level: int = logging.INFO) -> None:
    """Log dictionary in readable format. Used 6+ times.
    
    Args:
        logger: Logger instance
        data: Dictionary to log
        level: Logging level
    """
    for key, value in data.items():
        logger.log(level, f"  {key}: {value}")


def log_separator(logger: logging.Logger, char: str = '-', length: int = 50) -> None:
    """Log a separator line. Used 10+ times.
    
    Args:
        logger: Logger instance
        char: Character to use for separator
        length: Length of separator
    """
    logger.info(char * length)


def log_section(logger: logging.Logger, title: str) -> None:
    """Log a section header. Used 15+ times.
    
    Args:
        logger: Logger instance
        title: Section title
    """
    logger.info("")
    logger.info("=" * 50)
    logger.info(f" {title}")
    logger.info("=" * 50)


def suppress_warnings() -> None:
    """Suppress common warnings. Used 4+ times in notebooks."""
    import warnings
    warnings.filterwarnings('ignore')
    
    # Suppress specific library warnings
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)