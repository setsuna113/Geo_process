"""File handler with rotation support."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from ..formatters import JsonFormatter


class FileHandler(RotatingFileHandler):
    """Enhanced file handler with automatic rotation and JSON formatting.
    
    Features:
    - Automatic directory creation
    - Size-based rotation
    - JSON formatting for machine parsing
    - Configurable backup count
    """
    
    def __init__(self,
                 filename: str,
                 max_bytes: int = 100 * 1024 * 1024,  # 100MB default
                 backup_count: int = 5,
                 encoding: str = 'utf-8',
                 use_json: bool = True):
        """Initialize file handler.
        
        Args:
            filename: Log file path
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
            encoding: File encoding
            use_json: Whether to use JSON formatting
        """
        # Ensure directory exists
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize parent
        super().__init__(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding
        )
        
        # Set formatter
        if use_json:
            self.setFormatter(JsonFormatter())
        else:
            # Use default formatter
            self.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        # Set default level
        self.setLevel(logging.DEBUG)