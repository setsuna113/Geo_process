# src/abstractions/types/processing_types.py
"""Processing-related type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
import time


class ProcessingStatus(Enum):
    """Status of processing operations."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ProcessingResult:
    """Standard result format for all processors."""
    success: bool
    items_processed: int
    items_failed: int
    elapsed_time: float
    memory_used_mb: float
    results: Optional[List[Any]] = None
    errors: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProcessorConfig:
    """Configuration object for processors."""
    batch_size: int = 1000
    max_workers: Optional[int] = None
    store_results: bool = True
    memory_limit_mb: Optional[int] = None
    tile_size: Optional[int] = None
    supports_chunking: bool = True
    enable_progress: bool = True
    enable_checkpoints: bool = True
    checkpoint_interval: int = 100
    timeout_seconds: Optional[float] = None
    
    @classmethod
    def from_config(cls, config_dict: Optional[Dict[str, Any]] = None, processor_name: str = "") -> 'ProcessorConfig':
        """Create ProcessorConfig from config dictionary."""
        if not config_dict:
            return cls()
        
        # Get processor-specific config
        processor_config = config_dict.get(f'processors.{processor_name}', {})
        global_processor_config = config_dict.get('processors', {})
        
        # Merge configs with processor-specific taking precedence
        merged_config = {**global_processor_config, **processor_config}
        
        # Map config keys to ProcessorConfig fields
        return cls(
            batch_size=merged_config.get('batch_size', 1000),
            max_workers=merged_config.get('max_workers'),
            store_results=merged_config.get('store_results', True),
            memory_limit_mb=merged_config.get('memory_limit_mb'),
            tile_size=merged_config.get('tile_size'),
            supports_chunking=merged_config.get('supports_chunking', True),
            enable_progress=merged_config.get('enable_progress', True),
            enable_checkpoints=merged_config.get('enable_checkpoints', True),
            checkpoint_interval=merged_config.get('checkpoint_interval', 100),
            timeout_seconds=merged_config.get('timeout_seconds')
        )


@dataclass
class TileProgress:
    """Progress information for tile processing."""
    tile_id: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    progress_percentage: float = 0.0
    memory_usage_mb: float = 0.0
    
    @property
    def processing_time_seconds(self) -> Optional[float]:
        """Get processing time in seconds."""
        if self.start_time is None:
            return None
        end_time = self.end_time or time.time()
        return end_time - self.start_time