"""Console handler with colored output for interactive use."""

import logging
import sys
from typing import Optional

from ..formatters import HumanFormatter


class ConsoleHandler(logging.StreamHandler):
    """Enhanced console handler with automatic color detection and formatting.
    
    Features:
    - Automatic TTY detection for color support
    - Human-readable formatting
    - Error stream separation
    """
    
    def __init__(self, 
                 stream=None,
                 use_colors: Optional[bool] = None,
                 show_context: bool = True):
        """Initialize console handler.
        
        Args:
            stream: Output stream (defaults to stderr)
            use_colors: Force color on/off (auto-detect if None)
            show_context: Whether to show context information
        """
        # Default to stderr for logging
        if stream is None:
            stream = sys.stderr
            
        super().__init__(stream)
        
        # Auto-detect color support if not specified
        if use_colors is None:
            use_colors = self._supports_color(stream)
        
        # Set human-readable formatter
        formatter = HumanFormatter(
            use_colors=use_colors,
            show_context=show_context
        )
        self.setFormatter(formatter)
        
        # Set default level
        self.setLevel(logging.INFO)
    
    def _supports_color(self, stream) -> bool:
        """Check if stream supports ANSI colors.
        
        Args:
            stream: Output stream to check
            
        Returns:
            True if colors are supported
        """
        # Check if stream is a TTY
        if not hasattr(stream, 'isatty') or not stream.isatty():
            return False
        
        # Check for common environment variables
        import os
        
        # Check for NO_COLOR standard
        if os.environ.get('NO_COLOR'):
            return False
            
        # Check for TERM
        term = os.environ.get('TERM', '')
        if term == 'dumb':
            return False
            
        # Check for common CI environments that support color
        if any(os.environ.get(var) for var in ['CI', 'GITHUB_ACTIONS', 'GITLAB_CI']):
            return True
            
        # Default to True for TTY
        return True
    
    def emit(self, record: logging.LogRecord):
        """Emit a record with special handling for errors.
        
        Args:
            record: LogRecord to emit
        """
        # Could add special handling here if needed
        super().emit(record)