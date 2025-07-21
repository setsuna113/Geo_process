# src/raster_data/loaders/base_loader.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Iterator, Union
from pathlib import Path
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
import logging

from src.config.config import Config

logger = logging.getLogger(__name__)

@dataclass
class RasterWindow:
    """Represents a window into a raster."""
    col_off: int
    row_off: int
    width: int
    height: int
    
    @property
    def slice(self) -> Tuple[slice, slice]:
        """Get numpy slice for this window."""
        return (slice(self.row_off, self.row_off + self.height),
                slice(self.col_off, self.col_off + self.width))

@dataclass
class RasterMetadata:
    """Metadata for a raster dataset."""
    width: int
    height: int
    bounds: Tuple[float, float, float, float]  # (west, south, east, north)
    crs: str
    pixel_size: Tuple[float, float]  # (x_size, y_size)
    data_type: str
    nodata_value: Optional[float]
    band_count: int
    
    @property
    def resolution_degrees(self) -> float:
        """Get resolution in degrees (assuming square pixels)."""
        return abs(self.pixel_size[0])

class BaseRasterLoader(ABC):
    """Base class for raster data loaders with lazy loading support."""
    
    def __init__(self, config: Config):
        self.config = config
        # Handle both dict-style and object-style config access
        raster_config = config.get('raster_processing', {}) if hasattr(config, 'get') else getattr(config, 'raster_processing', {})
        self.tile_size = getattr(raster_config, 'tile_size', 1000) if hasattr(raster_config, 'tile_size') else raster_config.get('tile_size', 1000)
        self.memory_limit_mb = getattr(raster_config, 'memory_limit_mb', 1000) if hasattr(raster_config, 'memory_limit_mb') else raster_config.get('memory_limit_mb', 1000)
        
        # For cache size, try multiple config paths
        if hasattr(config, 'lazy_loading'):
            lazy_config = getattr(config, 'lazy_loading', {})
            self.cache_size = getattr(lazy_config, 'chunk_size_mb', 100) if hasattr(lazy_config, 'chunk_size_mb') else lazy_config.get('chunk_size_mb', 100)
        else:
            self.cache_size = config.get('cache_size_mb', 100) if hasattr(config, 'get') else 100
            
        self._tile_cache: Dict[str, Any] = {}  # Simple cache, will be replaced with LRU
        
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        pass
    
    @abstractmethod
    def extract_metadata(self, file_path: Path) -> RasterMetadata:
        """Extract metadata without loading data."""
        pass
    
    @abstractmethod
    def _open_dataset(self, file_path: Path) -> Any:
        """Open the dataset for reading."""
        pass
    
    @abstractmethod
    def _read_window(self, dataset: Any, window: RasterWindow, band: int = 1) -> np.ndarray:
        """Read data from a specific window."""
        pass
    
    @abstractmethod
    def _close_dataset(self, dataset: Any) -> None:
        """Close the dataset."""
        pass
    
    @contextmanager
    def open_lazy(self, file_path: Path):
        """Open raster for lazy reading."""
        dataset = None
        try:
            dataset = self._open_dataset(file_path)
            yield LazyRasterReader(self, dataset, file_path)
        finally:
            if dataset is not None:
                self._close_dataset(dataset)
    
    def load_window(self, file_path: Path, bounds: Tuple[float, float, float, float], 
                   band: int = 1) -> np.ndarray:
        """Load data for a specific geographic window."""
        metadata = self.extract_metadata(file_path)
        
        # Convert geographic bounds to pixel coordinates
        window = self._bounds_to_window(bounds, metadata)
        
        with self.open_lazy(file_path) as reader:
            return reader.read_window(window, band)
    
    def iter_tiles(self, file_path: Path, band: int = 1) -> Iterator[Tuple[RasterWindow, np.ndarray]]:
        """Iterate over tiles of the raster."""
        metadata = self.extract_metadata(file_path)
        
        with self.open_lazy(file_path) as reader:
            for window in self._generate_tile_windows(metadata):
                data = reader.read_window(window, band)
                yield window, data
    
    def _bounds_to_window(self, bounds: Tuple[float, float, float, float], 
                         metadata: RasterMetadata) -> RasterWindow:
        """Convert geographic bounds to pixel window."""
        west, south, east, north = bounds
        raster_west, raster_south, raster_east, raster_north = metadata.bounds
        
        # Calculate pixel coordinates
        col_off = int((west - raster_west) / metadata.pixel_size[0])
        row_off = int((raster_north - north) / abs(metadata.pixel_size[1]))
        col_end = int((east - raster_west) / metadata.pixel_size[0])
        row_end = int((raster_north - south) / abs(metadata.pixel_size[1]))
        
        # Clip to raster bounds
        col_off = max(0, col_off)
        row_off = max(0, row_off)
        col_end = min(metadata.width, col_end)
        row_end = min(metadata.height, row_end)
        
        return RasterWindow(
            col_off=col_off,
            row_off=row_off,
            width=col_end - col_off,
            height=row_end - row_off
        )
    
    def _generate_tile_windows(self, metadata: RasterMetadata) -> Iterator[RasterWindow]:
        """Generate tile windows for processing entire raster."""
        for row in range(0, metadata.height, self.tile_size):
            for col in range(0, metadata.width, self.tile_size):
                width = min(self.tile_size, metadata.width - col)
                height = min(self.tile_size, metadata.height - row)
                
                yield RasterWindow(
                    col_off=col,
                    row_off=row,
                    width=width,
                    height=height
                )
    
    def estimate_memory_usage(self, window: RasterWindow, dtype: np.dtype) -> float:
        """Estimate memory usage in MB for a window."""
        bytes_per_pixel = dtype.itemsize
        total_pixels = window.width * window.height
        return (total_pixels * bytes_per_pixel) / (1024 * 1024)

class LazyRasterReader:
    """Lazy reader for raster data."""
    
    def __init__(self, loader: BaseRasterLoader, dataset: Any, file_path: Path):
        self.loader = loader
        self.dataset = dataset
        self.file_path = file_path
        self.metadata = loader.extract_metadata(file_path)
        
    def read_window(self, window: RasterWindow, band: int = 1) -> np.ndarray:
        """Read data from a window."""
        return self.loader._read_window(self.dataset, window, band)
    
    def read_point(self, x: float, y: float, band: int = 1) -> Optional[float]:
        """Read value at a specific coordinate."""
        # Convert to pixel coordinates
        col = int((x - self.metadata.bounds[0]) / self.metadata.pixel_size[0])
        row = int((self.metadata.bounds[3] - y) / abs(self.metadata.pixel_size[1]))
        
        # Check bounds
        if col < 0 or col >= self.metadata.width or row < 0 or row >= self.metadata.height:
            return None
        
        # Read single pixel
        window = RasterWindow(col, row, 1, 1)
        value = self.read_window(window, band)[0, 0]
        
        return None if value == self.metadata.nodata_value else float(value)