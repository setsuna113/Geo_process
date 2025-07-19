"""Base class for raster data sources with lazy loading and tile support."""

from abc import ABC, abstractmethod
from typing import Optional, Iterator, Tuple, Any, Dict, Union, List
from contextlib import contextmanager
import numpy as np
from pathlib import Path
import logging

from .lazy_loadable import LazyLoadable
from .tileable import Tileable
from .cacheable import Cacheable

logger = logging.getLogger(__name__)


class RasterTile:
    """Represents a single raster tile with data and metadata."""
    
    def __init__(self, 
                 data: np.ndarray,
                 bounds: Tuple[float, float, float, float],
                 tile_id: str,
                 crs: str = "EPSG:4326",
                 nodata: Optional[Union[int, float]] = None):
        """
        Initialize raster tile.
        
        Args:
            data: Numpy array containing tile data
            bounds: Tile bounds (minx, miny, maxx, maxy)
            tile_id: Unique tile identifier
            crs: Coordinate reference system
            nodata: No-data value
        """
        self.data = data
        self.bounds = bounds
        self.tile_id = tile_id
        self.crs = crs
        self.nodata = nodata
        self.shape = data.shape
        self.dtype = data.dtype
        
    @property
    def memory_size_mb(self) -> float:
        """Calculate memory size of tile in MB."""
        return self.data.nbytes / (1024 * 1024)
        
    def is_empty(self) -> bool:
        """Check if tile contains only no-data values."""
        if self.nodata is None:
            return False
        return bool(np.all(self.data == self.nodata))


class BaseRasterSource(LazyLoadable, Tileable, Cacheable, ABC):
    """
    Abstract base class for raster data sources.
    
    Provides interface for lazy loading, tile iteration, and memory management.
    """
    
    def __init__(self, 
                 source_path: Union[str, Path],
                 bands: Optional[List[int]] = None,
                 tile_size: int = 512,
                 cache_tiles: bool = True,
                 **kwargs):
        """
        Initialize raster source.
        
        Args:
            source_path: Path to raster data source
            bands: List of band indices to read (1-based)
            tile_size: Default tile size in pixels
            cache_tiles: Whether to cache tiles in memory
            **kwargs: Additional source-specific parameters
        """
        super().__init__()
        
        self.source_path = Path(source_path)
        self.bands = bands or [1]
        self.tile_size = tile_size
        self.cache_tiles = cache_tiles
        
        # Will be set during initialization
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._band_count: Optional[int] = None
        self._dtype: Optional[np.dtype] = None
        self._crs: Optional[str] = None
        self._transform: Optional[Any] = None  # Affine transform
        self._nodata: Optional[Union[int, float]] = None
        self._bounds: Optional[Tuple[float, float, float, float]] = None
        
        # Initialize source-specific properties
        self._initialize_source()
        
    @abstractmethod
    def _initialize_source(self) -> None:
        """Initialize source-specific properties (width, height, etc.)."""
        pass
        
    @abstractmethod
    def _read_tile_data(self, 
                       window: Tuple[int, int, int, int],
                       bands: Optional[List[int]] = None) -> np.ndarray:
        """
        Read data for a specific window.
        
        Args:
            window: (col_off, row_off, width, height) in pixels
            bands: Band indices to read
            
        Returns:
            Numpy array with shape (bands, height, width)
        """
        pass
        
    @abstractmethod
    def _get_window_bounds(self, window: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        """Get geographic bounds for a pixel window."""
        pass
        
    @property
    def width(self) -> int:
        """Raster width in pixels."""
        if self._width is None:
            raise ValueError("Raster source not properly initialized")
        return self._width
        
    @property
    def height(self) -> int:
        """Raster height in pixels."""
        if self._height is None:
            raise ValueError("Raster source not properly initialized")
        return self._height
        
    @property
    def band_count(self) -> int:
        """Number of bands in the raster."""
        if self._band_count is None:
            raise ValueError("Raster source not properly initialized")
        return self._band_count
        
    @property
    def dtype(self) -> np.dtype:
        """Data type of raster values."""
        if self._dtype is None:
            raise ValueError("Raster source not properly initialized")
        return self._dtype
        
    @property
    def crs(self) -> str:
        """Coordinate reference system."""
        if self._crs is None:
            raise ValueError("Raster source not properly initialized")
        return self._crs
        
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Geographic bounds (minx, miny, maxx, maxy)."""
        if self._bounds is None:
            raise ValueError("Raster source not properly initialized")
        return self._bounds
        
    @property
    def nodata(self) -> Optional[Union[int, float]]:
        """No-data value."""
        return self._nodata
        
    @property
    def resolution(self) -> Tuple[float, float]:
        """Pixel resolution (x_res, y_res) in CRS units."""
        if self._transform is None:
            raise ValueError("Transform not available")
        # Assuming Affine transform with scale in [0] and [4]
        return (abs(self._transform[0]), abs(self._transform[4]))
        
    def estimate_memory_usage(self, 
                            window: Optional[Tuple[int, int, int, int]] = None,
                            bands: Optional[List[int]] = None) -> float:
        """
        Estimate memory usage in MB for reading data.
        
        Args:
            window: Pixel window to read, defaults to full raster
            bands: Bands to read, defaults to self.bands
            
        Returns:
            Estimated memory usage in MB
        """
        if window is None:
            width, height = self.width, self.height
        else:
            width, height = window[2], window[3]
            
        bands = bands or self.bands
        band_count = len(bands)
        
        # Calculate size based on data type
        dtype_size = np.dtype(self.dtype).itemsize
        total_bytes = width * height * band_count * dtype_size
        
        return total_bytes / (1024 * 1024)
        
    def get_tile_iterator(self, 
                         tile_size: Optional[int] = None,
                         overlap: int = 0,
                         bands: Optional[List[int]] = None) -> Iterator[RasterTile]:
        """
        Get iterator over raster tiles.
        
        Args:
            tile_size: Size of tiles in pixels
            overlap: Overlap between tiles in pixels
            bands: Bands to include in tiles
            
        Yields:
            RasterTile objects
        """
        tile_size = tile_size or self.tile_size
        bands = bands or self.bands
        
        # Calculate tile grid
        cols = range(0, self.width, tile_size - overlap)
        rows = range(0, self.height, tile_size - overlap)
        
        for tile_row, row_start in enumerate(rows):
            for tile_col, col_start in enumerate(cols):
                # Calculate actual window size
                col_end = min(col_start + tile_size, self.width)
                row_end = min(row_start + tile_size, self.height)
                
                window = (col_start, row_start, col_end - col_start, row_end - row_start)
                tile_id = f"tile_{tile_row}_{tile_col}"
                
                # Check cache first if enabled
                if self.cache_tiles:
                    cached_tile = self.get_cached(tile_id)
                    if cached_tile is not None:
                        yield cached_tile
                        continue
                
                # Read tile data
                try:
                    data = self._read_tile_data(window, bands)
                    bounds = self._get_window_bounds(window)
                    
                    tile = RasterTile(
                        data=data,
                        bounds=bounds,
                        tile_id=tile_id,
                        crs=self.crs,
                        nodata=self.nodata
                    )
                    
                    # Cache tile if enabled
                    if self.cache_tiles:
                        self.cache(tile_id, tile)
                    
                    yield tile
                    
                except Exception as e:
                    logger.error(f"Failed to read tile {tile_id}: {e}")
                    continue
                    
    @contextmanager
    def open_context(self):
        """Context manager for resource management."""
        try:
            self._open_resource()
            yield self
        finally:
            self._close_resource()
            
    def _open_resource(self) -> None:
        """Open the raster resource for reading."""
        # Override in subclasses if needed
        pass
        
    def _close_resource(self) -> None:
        """Close the raster resource."""
        # Override in subclasses if needed
        pass
        
    def get_optimal_tile_size(self, memory_limit_mb: float = 512) -> int:
        """
        Calculate optimal tile size based on memory constraints.
        
        Args:
            memory_limit_mb: Maximum memory per tile in MB
            
        Returns:
            Optimal tile size in pixels
        """
        band_count = len(self.bands)
        dtype_size = np.dtype(self.dtype).itemsize
        
        # Calculate pixels per MB
        bytes_per_mb = 1024 * 1024
        pixels_per_mb = bytes_per_mb / (band_count * dtype_size)
        
        # Calculate max pixels for memory limit
        max_pixels = int(pixels_per_mb * memory_limit_mb)
        
        # Return square root for square tiles
        optimal_size = int(np.sqrt(max_pixels))
        
        # Ensure it's a reasonable size (between 64 and 2048)
        return max(64, min(optimal_size, 2048))
        
    def supports_random_access(self) -> bool:
        """Check if source supports efficient random tile access."""
        # Override in subclasses - default to True
        return True
        
    def get_overview_levels(self) -> List[int]:
        """Get available overview/pyramid levels."""
        # Override in subclasses if overviews are supported
        return []
        
    def read_at_resolution(self, 
                          target_resolution: Tuple[float, float],
                          window: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Read data at a specific resolution using overviews if available.
        
        Args:
            target_resolution: Desired (x_res, y_res) in CRS units
            window: Pixel window in original resolution
            
        Returns:
            Resampled data array
        """
        # Default implementation - override in subclasses for efficiency
        current_res = self.resolution
        scale_x = current_res[0] / target_resolution[0]
        scale_y = current_res[1] / target_resolution[1]
        
        if abs(scale_x - 1.0) < 0.01 and abs(scale_y - 1.0) < 0.01:
            # No resampling needed
            return self._read_tile_data(window or (0, 0, self.width, self.height))
        
        # Simple resampling - override for better algorithms
        data = self._read_tile_data(window or (0, 0, self.width, self.height))
        
        if len(data.shape) == 3:  # Multi-band
            new_height = int(data.shape[1] / scale_y)
            new_width = int(data.shape[2] / scale_x)
            
            # Simple nearest neighbor - replace with proper resampling
            step_y = max(1, int(scale_y))
            step_x = max(1, int(scale_x))
            
            return data[:, ::step_y, ::step_x]
        else:  # Single band
            new_height = int(data.shape[0] / scale_y)
            new_width = int(data.shape[1] / scale_x)
            
            step_y = max(1, int(scale_y))
            step_x = max(1, int(scale_x))
            
            return data[::step_y, ::step_x]
