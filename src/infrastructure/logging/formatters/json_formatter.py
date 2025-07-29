"""JSON formatter for structured machine-readable logs."""

import logging
import json
from datetime import datetime
from typing import Dict, Any


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for machine parsing.
    
    Produces single-line JSON records with all context and metadata.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: LogRecord to format
            
        Returns:
            JSON string (single line)
        """
        # Base log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process
        }
        
        # Add context if present
        if hasattr(record, 'context'):
            context = getattr(record, 'context', {})
            if context:
                log_data['context'] = context
        
        # Add performance data if present
        if hasattr(record, 'performance'):
            perf = getattr(record, 'performance')
            if perf:
                log_data['performance'] = perf
        
        # Add traceback if present
        if hasattr(record, 'traceback'):
            tb = getattr(record, 'traceback')
            if tb:
                log_data['traceback'] = tb
        elif record.exc_info:
            log_data['traceback'] = self.formatException(record.exc_info)
        
        # Convert to JSON (single line for log aggregation)
        return json.dumps(log_data, separators=(',', ':'), default=str)
    
    def formatException(self, exc_info) -> str:
        """Format exception info as string.
        
        Args:
            exc_info: Exception info tuple
            
        Returns:
            Formatted traceback string
        """
        import traceback
        return ''.join(traceback.format_exception(*exc_info))


class PrettyJsonFormatter(JsonFormatter):
    """JSON formatter with pretty printing for human readability."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as pretty-printed JSON.
        
        Args:
            record: LogRecord to format
            
        Returns:
            Pretty-printed JSON string
        """
        # Get base JSON data
        log_data = json.loads(super().format(record))
        
        # Pretty print with indentation
        return json.dumps(log_data, indent=2, default=str)