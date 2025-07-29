"""Human-readable formatter for console output."""

import logging
from datetime import datetime
from typing import Dict, Optional


class HumanFormatter(logging.Formatter):
    """Format log records for human readability with colors and structure.
    
    Features:
    - Color coding by log level
    - Contextual information display
    - Compact format for readability
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[0m',        # Default
        'WARNING': '\033[93m',    # Yellow
        'ERROR': '\033[91m',      # Red
        'CRITICAL': '\033[95m',   # Magenta
    }
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    def __init__(self, use_colors: bool = True, show_context: bool = True):
        """Initialize formatter.
        
        Args:
            use_colors: Whether to use ANSI colors
            show_context: Whether to show context information
        """
        super().__init__()
        self.use_colors = use_colors
        self.show_context = show_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human reading.
        
        Args:
            record: LogRecord to format
            
        Returns:
            Formatted string with optional colors
        """
        # Get timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Get color for level
        if self.use_colors:
            level_color = self.COLORS.get(record.levelname, '')
            reset = self.RESET
            bold = self.BOLD
            dim = self.DIM
        else:
            level_color = reset = bold = dim = ''
        
        # Format level (fixed width)
        level = f"{level_color}{record.levelname:8}{reset}"
        
        # Get context information
        context_str = self._format_context(record) if self.show_context else ''
        
        # Get logger name (shortened)
        logger_name = self._shorten_logger_name(record.name)
        
        # Base message
        message = record.getMessage()
        
        # Build formatted output
        parts = [
            f"{dim}{timestamp}{reset}",
            level,
            f"{dim}[{logger_name}]{reset}",
        ]
        
        if context_str:
            parts.append(f"{bold}{context_str}{reset}")
            
        parts.append(message)
        
        output = ' '.join(parts)
        
        # Add performance metrics if present
        if hasattr(record, 'performance') and record.performance:
            perf = record.performance
            perf_str = self._format_performance(perf)
            if perf_str:
                output += f"\n  {dim}Performance: {perf_str}{reset}"
        
        # Add traceback if present
        if hasattr(record, 'traceback') and record.traceback:
            tb = record.traceback
            if self.use_colors:
                tb_lines = tb.strip().split('\n')
                tb_colored = '\n'.join(f"  {level_color}{line}{reset}" for line in tb_lines)
                output += f"\n{tb_colored}"
            else:
                output += f"\n{tb}"
        
        return output
    
    def _format_context(self, record: logging.LogRecord) -> str:
        """Format context information.
        
        Args:
            record: LogRecord with context
            
        Returns:
            Formatted context string
        """
        if not hasattr(record, 'context'):
            return ''
            
        context = record.context
        parts = []
        
        # Key context fields to show
        if context.get('experiment_id'):
            # Show first 8 chars of UUID
            exp_id = str(context['experiment_id'])[:8]
            parts.append(f"exp:{exp_id}")
            
        if context.get('stage'):
            parts.append(f"stage:{context['stage']}")
            
        if context.get('node_id'):
            # Show last component of node path
            node = context['node_id'].split('/')[-1]
            parts.append(f"node:{node}")
            
        return f"[{' | '.join(parts)}]" if parts else ''
    
    def _shorten_logger_name(self, name: str, max_length: int = 20) -> str:
        """Shorten logger name for display.
        
        Args:
            name: Full logger name
            max_length: Maximum length
            
        Returns:
            Shortened name
        """
        if len(name) <= max_length:
            return name
            
        # Try to show last component
        parts = name.split('.')
        if len(parts) > 1:
            last = parts[-1]
            if len(last) <= max_length - 3:
                return f"...{last}"
        
        # Just truncate
        return f"{name[:max_length-3]}..."
    
    def _format_performance(self, perf: Dict) -> str:
        """Format performance metrics.
        
        Args:
            perf: Performance data dict
            
        Returns:
            Formatted performance string
        """
        parts = []
        
        if 'duration_seconds' in perf:
            parts.append(f"{perf['duration_seconds']:.3f}s")
            
        if 'items_per_second' in perf:
            parts.append(f"{perf['items_per_second']:.1f} items/s")
            
        if 'memory_mb' in perf:
            parts.append(f"{perf['memory_mb']:.1f} MB")
            
        return ' | '.join(parts)