# src/processors/exporters/base_exporter.py
"""Base exporter for data export operations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterator, List
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ExportConfig:
    """Configuration for export operations."""
    
    def __init__(self, 
                 output_path: Path,
                 chunk_size: int = 10000,
                 include_metadata: bool = True,
                 compression: Optional[str] = None,
                 **kwargs):
        self.output_path = Path(output_path)
        self.chunk_size = chunk_size
        self.include_metadata = include_metadata
        self.compression = compression
        self.additional_options = kwargs


class BaseExporter(ABC):
    """Abstract base class for data exporters."""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.export_stats = {
            'rows_exported': 0,
            'chunks_processed': 0,
            'start_time': None,
            'end_time': None
        }
    
    @abstractmethod
    def export(self, 
               dataset_info: Dict[str, Any],
               config: ExportConfig,
               progress_callback: Optional[callable] = None) -> Path:
        """
        Export data to specified format.
        
        Args:
            dataset_info: Information about dataset to export
            config: Export configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to exported file
        """
        pass
    
    @abstractmethod
    def validate_export(self, output_path: Path) -> bool:
        """Validate the exported file."""
        pass
    
    def get_export_stats(self) -> Dict[str, Any]:
        """Get export statistics."""
        stats = self.export_stats.copy()
        if stats['start_time'] and stats['end_time']:
            stats['duration_seconds'] = (
                stats['end_time'] - stats['start_time']
            ).total_seconds()
        return stats