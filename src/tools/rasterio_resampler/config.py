"""Configuration for rasterio resampler with system resource detection."""

import os
import psutil
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResamplingConfig:
    """Configuration for rasterio resampling with adaptive resource limits."""
    
    # Target resolution and CRS
    target_resolution: float = 0.166744  # ~18.5km at equator, matching config.yml
    target_crs: str = 'EPSG:4326'
    
    # Resampling algorithm
    resampling_method: str = 'average'  # average, bilinear, cubic, etc.
    
    # Memory management
    memory_limit_gb: Optional[float] = None  # Auto-detect if None
    memory_safety_factor: float = 0.8  # Use 80% of available memory
    window_size: int = 2048  # Process in chunks
    
    # CPU management
    max_workers: Optional[int] = None  # Auto-detect if None
    cpu_safety_factor: float = 0.75  # Use 75% of available CPUs
    
    # Progress tracking
    checkpoint_interval: int = 10  # Save progress every N windows
    progress_file: str = "resampling_progress.json"
    
    # Debugging
    debug: bool = False
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Output options
    output_dir: str = "./resampled"
    compress: str = 'lzw'
    tiled: bool = True
    blockxsize: int = 512
    blockysize: int = 512
    
    # Background running
    daemon: bool = False
    pid_file: Optional[str] = None
    
    def __post_init__(self):
        """Auto-detect system resources and set limits."""
        # Auto-detect memory limit
        if self.memory_limit_gb is None:
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            self.memory_limit_gb = total_memory_gb * self.memory_safety_factor
            logger.info(f"Auto-detected memory limit: {self.memory_limit_gb:.1f} GB")
        
        # Auto-detect CPU limit
        if self.max_workers is None:
            cpu_count = psutil.cpu_count()
            self.max_workers = max(1, int(cpu_count * self.cpu_safety_factor))
            logger.info(f"Auto-detected max workers: {self.max_workers}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_memory_limit_bytes(self) -> int:
        """Get memory limit in bytes."""
        return int(self.memory_limit_gb * 1024**3)
    
    def estimate_window_memory(self, width: int, height: int, dtype_size: int = 4) -> int:
        """Estimate memory needed for a window in bytes."""
        # Account for input, output, and overhead
        overhead_factor = 3  # Input + output + working memory
        return width * height * dtype_size * overhead_factor
    
    def get_optimal_window_size(self, raster_width: int, raster_height: int, 
                                dtype_size: int = 4) -> int:
        """Calculate optimal window size based on available memory.
        
        Simple, conservative approach:
        - Use 25% of available memory (very conservative)
        - Divide equally among workers
        - Fixed overhead factor of 3x (input + output + working memory)
        """
        # Very conservative: use only 25% of available memory
        available_memory = self.get_memory_limit_bytes() * 0.25
        
        # Simple division among workers
        memory_per_worker = available_memory / max(1, self.max_workers)
        
        # Fixed overhead factor: 3x for input, output, and working memory
        overhead_factor = 3
        window_pixels = memory_per_worker / (dtype_size * overhead_factor)
        
        # Calculate square window size
        window_size = int((window_pixels ** 0.5))
        
        # Constrain to reasonable bounds
        min_window = 256  # Minimum for efficiency
        max_window = min(self.window_size, raster_width, raster_height)
        window_size = max(min_window, min(window_size, max_window))
        
        # Align to block size
        if window_size > self.blockxsize:
            window_size = (window_size // self.blockxsize) * self.blockxsize
        
        return window_size
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ResamplingConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() 
                     if k in cls.__dataclass_fields__})