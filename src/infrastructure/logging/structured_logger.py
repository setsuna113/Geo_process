"""Structured logging with context propagation for pipeline monitoring."""

import logging
import sys
import json
import traceback
from typing import Dict, Any, Optional, Union
from contextvars import ContextVar
from datetime import datetime
import time

# Context variables for correlation across async operations
experiment_context: ContextVar[Optional[str]] = ContextVar('experiment_id', default=None)
node_context: ContextVar[Optional[str]] = ContextVar('node_id', default=None)
stage_context: ContextVar[Optional[str]] = ContextVar('stage', default=None)
job_context: ContextVar[Optional[str]] = ContextVar('job_id', default=None)


class StructuredLogger(logging.Logger):
    """Enhanced logger with structured output and context propagation.
    
    Features:
    - Automatic context injection (experiment_id, node_id, stage)
    - Structured JSON output for machine parsing
    - Performance metrics logging
    - Full traceback capture for errors
    - Correlation IDs across distributed operations
    """
    
    def __init__(self, name: str):
        """Initialize structured logger.
        
        Args:
            name: Logger name (usually __name__)
        """
        super().__init__(name)
        self._context_fields: Dict[str, Any] = {}
        self._start_times: Dict[str, float] = {}
    
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, **kwargs):
        """Override to add context and structure.
        
        Enhances log records with:
        - Context from ContextVars
        - Structured fields
        - Traceback capture
        - Performance metrics
        """
        # Gather context from ContextVars
        context = {
            'experiment_id': experiment_context.get(),
            'job_id': job_context.get(),
            'node_id': node_context.get(),
            'stage': stage_context.get(),
            'logger_name': self.name,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            **self._context_fields  # Persistent context fields
        }
        
        # Remove None values
        context = {k: v for k, v in context.items() if v is not None}
        
        # Add any extra context from caller
        if extra and isinstance(extra, dict):
            # Extract special fields
            performance = extra.pop('performance', None)
            extra_context = extra.pop('context', {})
            traceback_str = extra.pop('traceback', None)
            
            # Merge contexts
            context.update(extra_context)
        else:
            performance = None
            traceback_str = None
        
        # Capture traceback if error
        if not traceback_str and exc_info:
            if isinstance(exc_info, bool):
                exc_info = sys.exc_info()
            if exc_info[0] is not None:
                traceback_str = ''.join(traceback.format_exception(*exc_info))
        
        # Create structured extra data
        structured_extra = {
            'context': context,
            'performance': performance,
            'traceback': traceback_str
        }
        
        # Update extra dict
        if extra is None:
            extra = {}
        extra.update(structured_extra)
        
        # Call parent with structured data
        super()._log(level, msg, args, exc_info=False, extra=extra, 
                    stack_info=stack_info, **kwargs)
    
    def add_context(self, **fields):
        """Add persistent context fields to all future log messages.
        
        Args:
            **fields: Key-value pairs to add to context
            
        Example:
            logger.add_context(user_id='123', request_id='abc')
        """
        self._context_fields.update(fields)
    
    def remove_context(self, *keys):
        """Remove persistent context fields.
        
        Args:
            *keys: Keys to remove from context
        """
        for key in keys:
            self._context_fields.pop(key, None)
    
    def clear_context(self):
        """Clear all persistent context fields."""
        self._context_fields.clear()
    
    def start_operation(self, operation: str):
        """Start timing an operation.
        
        Args:
            operation: Operation name for timing
        """
        self._start_times[operation] = time.time()
        self.debug(f"Started operation: {operation}")
    
    def end_operation(self, operation: str, **metrics):
        """End timing an operation and log performance.
        
        Args:
            operation: Operation name
            **metrics: Additional metrics to log
        """
        if operation not in self._start_times:
            self.warning(f"No start time for operation: {operation}")
            return
            
        duration = time.time() - self._start_times.pop(operation)
        self.log_performance(operation, duration, **metrics)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics for an operation.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            **metrics: Additional metrics (items_processed, memory_mb, etc.)
            
        Example:
            logger.log_performance('data_load', 1.23, 
                                 items_processed=1000,
                                 memory_mb=512)
        """
        performance_data = {
            'operation': operation,
            'duration_seconds': round(duration, 3),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            **metrics
        }
        
        # Calculate rate if items provided
        if 'items_processed' in metrics and duration > 0:
            performance_data['items_per_second'] = round(
                metrics['items_processed'] / duration, 2
            )
        
        self.info(
            f"Performance: {operation} completed in {duration:.3f}s",
            extra={'performance': performance_data}
        )
    
    def log_error_with_context(self, error: Exception, operation: str = None, **context):
        """Log an error with full context and traceback.
        
        Args:
            error: The exception that occurred
            operation: Optional operation name for context
            **context: Additional context fields
        """
        error_context = {
            'error_type': type(error).__name__,
            'error_module': type(error).__module__,
            **context
        }
        
        if operation:
            error_context['operation'] = operation
            
        self.error(
            f"{type(error).__name__}: {str(error)}",
            exc_info=True,
            extra={'context': error_context}
        )
    
    def create_child(self, suffix: str) -> 'StructuredLogger':
        """Create a child logger with additional name component.
        
        Args:
            suffix: Suffix to add to logger name
            
        Returns:
            Child StructuredLogger instance
        """
        child_name = f"{self.name}.{suffix}"
        return get_logger(child_name)


# Global logger cache
_logger_cache: Dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        StructuredLogger instance
        
    Example:
        from src.infrastructure.logging import get_logger
        logger = get_logger(__name__)
    """
    if name in _logger_cache:
        return _logger_cache[name]
    
    # Temporarily set logger class
    original_class = logging.getLoggerClass()
    logging.setLoggerClass(StructuredLogger)
    
    try:
        logger = logging.getLogger(name)
        _logger_cache[name] = logger
        return logger
    finally:
        # Restore original logger class
        logging.setLoggerClass(original_class)


def setup_daemon_logging(process_name: str, log_file: str, experiment_id: str = None):
    """Setup logging for daemon processes.
    
    Configures structured logging with file output for daemon processes
    that may lose stdout/stderr.
    
    Args:
        process_name: Name of the daemon process
        log_file: Path to log file
        experiment_id: Optional experiment ID for context
    """
    from .handlers import FileHandler
    from .formatters import JsonFormatter
    
    # Set experiment context if provided
    if experiment_id:
        experiment_context.set(experiment_id)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add structured file handler
    file_handler = FileHandler(log_file)
    file_handler.setFormatter(JsonFormatter())
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    # Log startup
    logger = get_logger('daemon')
    logger.info(
        f"Daemon process started: {process_name}",
        extra={'context': {'process_name': process_name}}
    )