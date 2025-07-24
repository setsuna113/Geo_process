# src/config/processing_config.py
"""Configuration classes for memory-aware processing."""

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any


@dataclass
class ProcessingConfig:
    """
    Configuration for memory-aware processing.
    
    This configuration is used by various components (stages, processors, engines)
    to control memory usage and processing behavior.
    """
    chunk_size: Optional[int] = None  # Size of data chunks to process
    memory_limit_mb: float = 4096  # Memory limit in MB
    enable_chunking: bool = True  # Enable chunked processing
    checkpoint_interval: int = 100  # Checkpoint every N chunks
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    
    # Additional processing options
    use_memory_mapping: bool = False  # Use memory-mapped files where possible
    compression_level: int = 4  # Compression level for intermediate files (0-9)
    temp_dir: Optional[str] = None  # Directory for temporary files
    cleanup_temp_files: bool = True  # Automatically cleanup temporary files
    
    # Performance tuning
    parallel_chunks: int = 1  # Number of chunks to process in parallel
    prefetch_chunks: int = 1  # Number of chunks to prefetch
    
    # Retry configuration
    retry_on_memory_error: bool = True  # Automatically retry with smaller chunks
    min_chunk_size: int = 100  # Minimum chunk size before giving up
    chunk_size_reduction_factor: float = 0.5  # Reduce chunk size by this factor on retry
    
    def adjust_for_memory_pressure(self, available_memory_mb: float):
        """Adjust configuration based on available memory."""
        if available_memory_mb < self.memory_limit_mb * 0.5:
            # Less than 50% of desired memory available
            self.chunk_size = max(
                self.min_chunk_size,
                int((self.chunk_size or 1000) * self.chunk_size_reduction_factor)
            )
            self.parallel_chunks = 1
            self.prefetch_chunks = 0
            self.use_memory_mapping = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'chunk_size': self.chunk_size,
            'memory_limit_mb': self.memory_limit_mb,
            'enable_chunking': self.enable_chunking,
            'checkpoint_interval': self.checkpoint_interval,
            'use_memory_mapping': self.use_memory_mapping,
            'compression_level': self.compression_level,
            'temp_dir': self.temp_dir,
            'cleanup_temp_files': self.cleanup_temp_files,
            'parallel_chunks': self.parallel_chunks,
            'prefetch_chunks': self.prefetch_chunks,
            'retry_on_memory_error': self.retry_on_memory_error,
            'min_chunk_size': self.min_chunk_size,
            'chunk_size_reduction_factor': self.chunk_size_reduction_factor
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingConfig':
        """Create from dictionary."""
        # Filter out any keys that aren't in the dataclass
        valid_keys = {
            'chunk_size', 'memory_limit_mb', 'enable_chunking', 
            'checkpoint_interval', 'use_memory_mapping', 'compression_level',
            'temp_dir', 'cleanup_temp_files', 'parallel_chunks', 
            'prefetch_chunks', 'retry_on_memory_error', 'min_chunk_size',
            'chunk_size_reduction_factor'
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass 
class ChunkInfo:
    """Information about a data chunk being processed."""
    index: int  # Chunk index
    total_chunks: int  # Total number of chunks
    start_row: int  # Starting row in the full dataset
    end_row: int  # Ending row in the full dataset
    start_col: int  # Starting column in the full dataset
    end_col: int  # Ending column in the full dataset
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get chunk shape."""
        return (self.end_row - self.start_row, self.end_col - self.start_col)
    
    @property
    def progress_percent(self) -> float:
        """Get progress percentage."""
        return (self.index + 1) / self.total_chunks * 100


# Import Tuple for ChunkInfo
from typing import Tuple