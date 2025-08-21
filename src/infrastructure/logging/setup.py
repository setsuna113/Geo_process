"""Setup and configuration for the structured logging system."""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from .structured_logger import get_logger
from .handlers import DatabaseLogHandler, ConsoleHandler, FileHandler
from .formatters import JsonFormatter, HumanFormatter


def setup_logging(config: Dict[str, Any], 
                 db_manager = None,
                 experiment_id: Optional[str] = None,
                 log_file: Optional[str] = None,
                 console: bool = True,
                 database: bool = True,
                 log_level: str = 'INFO'):
    """Configure the structured logging system.
    
    Args:
        config: Configuration dict (from Config class)
        db_manager: DatabaseManager instance (required for database logging)
        experiment_id: Current experiment ID for context
        log_file: Optional log file path (uses config default if not provided)
        console: Whether to enable console logging
        database: Whether to enable database logging
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Get root logger
    root_logger = logging.getLogger()
    
    # Set level
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    if console:
        console_handler = ConsoleHandler(
            use_colors=sys.stderr.isatty(),
            show_context=True
        )
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        # Use config default
        log_dir = Path(config.get('paths.logs_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'pipeline.log'
    
    file_handler = FileHandler(
        filename=str(log_file),
        max_bytes=config.get('logging.max_file_size', 100 * 1024 * 1024),  # 100MB
        backup_count=config.get('logging.backup_count', 5),
        use_json=True
    )
    file_handler.setLevel(logging.DEBUG)  # Capture everything in files
    root_logger.addHandler(file_handler)
    
    # Database handler
    if database and db_manager:
        try:
            db_handler = DatabaseLogHandler(
                db_manager=db_manager,
                batch_size=config.get('monitoring.log_batch_size', 100),
                flush_interval=config.get('monitoring.log_flush_interval', 5.0),
                max_queue_size=config.get('monitoring.log_queue_size', 10000)
            )
            db_handler.setLevel(logging.INFO)  # Don't flood database with DEBUG
            root_logger.addHandler(db_handler)
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to setup database logging: {e}")
    
    # Set experiment context if provided
    if experiment_id:
        from .context import experiment_context
        experiment_context.set(experiment_id)
    
    # Log startup
    logger = get_logger(__name__)
    logger.info(
        "Structured logging system initialized",
        extra={
            'context': {
                'log_level': log_level,
                'handlers': {
                    'console': console,
                    'file': str(log_file),
                    'database': database and db_manager is not None
                }
            }
        }
    )


def setup_simple_logging(log_level: str = 'INFO'):
    """Setup basic logging without database for testing/debugging.
    
    Args:
        log_level: Minimum log level
    """
    # Get root logger
    root_logger = logging.getLogger()
    
    # Set level
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler only
    console_handler = ConsoleHandler(show_context=True)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)


def get_log_stats() -> Dict[str, Any]:
    """Get statistics from all log handlers.
    
    Returns:
        Dict with handler statistics
    """
    stats = {}
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, DatabaseLogHandler):
            stats['database'] = handler.get_stats()
        elif isinstance(handler, FileHandler):
            # Could add file size stats here
            stats['file'] = {
                'filename': handler.baseFilename,
                'max_bytes': handler.maxBytes,
                'backup_count': handler.backupCount
            }
    
    return stats