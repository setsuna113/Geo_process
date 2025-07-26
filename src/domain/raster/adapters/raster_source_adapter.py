# src/raster_data/adapters/raster_source_adapter.py
"""Adapter to bridge BaseRasterLoader with BaseRasterSource interface."""

from typing import Any, Iterator, Tuple, Optional, Union, List
from pathlib import Path
import numpy as np

from src.base.raster_source import BaseRasterSource, RasterTile
from src.domain.raster.loaders.base_loader import BaseRasterLoader, RasterWindow, RasterMetadata


class RasterSourceAdapter(BaseRasterSource):
    """
    Adapter that wraps a BaseRasterLoader to provide BaseRasterSource interface.
    
    This allows raster_data loaders to work with systems expecting BaseRasterSource
    while maintaining separation of concerns.
    """
    
    def __init__(self, loader: BaseRasterLoader, file_path: Path, **kwargs):
        """
        Initialize adapter with a loader.
        
        Args:
            loader: BaseRasterLoader instance to wrap
            file_path: Path to raster file
            **kwargs: Additional arguments for BaseRasterSource
        """
        self._loader = loader
        self._file_path = file_path
        self._metadata: Optional[RasterMetadata] = None
        
        # Initialize BaseRasterSource with file path
        super().__init__(source_path=file_path, **kwargs)
        
        # Initialize metadata immediately
        self._initialize_source()
        
    def _initialize_source(self) -> None:
        """Initialize source-specific properties."""
        # Extract metadata using the wrapped loader
        self._metadata = self._loader.extract_metadata(self._file_path)
        
        # Set BaseRasterSource properties from metadata
        self._width = self._metadata.width
        self._height = self._metadata.height
        self._band_count = self._metadata.band_count
        self._crs = self._metadata.crs
        self._bounds = self._metadata.bounds
        self._nodata = self._metadata.nodata_value
        
        # Set transform and dtype
        pixel_size_x, pixel_size_y = self._metadata.pixel_size
        # Simple transform (assumes no rotation/skew)
        self._transform = (
            self._metadata.bounds[0],  # x origin
            pixel_size_x,             # x pixel size
            0.0,                      # x skew
            self._metadata.bounds[3],  # y origin  
            0.0,                      # y skew
            pixel_size_y              # y pixel size (usually negative)
        )
        
        # Set dtype from metadata
        import numpy as np
        dtype_map = {
            'Int32': np.int32,
            'UInt16': np.uint16,
            'Float32': np.float32,
            'Float64': np.float64,
        }
        self._dtype: Optional[np.dtype] = np.dtype(dtype_map.get(self._metadata.data_type, np.float32))
    
    # Implement abstract methods from LazyLoadable
    def _load_resource(self) -> Any:
        """Load the raster resource."""
        # Return dataset handle or metadata
        return self._metadata
    
    def _unload_resource(self) -> None:
        """Unload the resource."""
        # BaseRasterLoader doesn't maintain persistent handles
        pass
    
    # Implement abstract methods from Tileable  
    def get_dimensions(self) -> Tuple[int, int]:
        """Get raster dimensions (width, height)."""
        if self._width is None or self._height is None:
            raise ValueError("Adapter not properly initialized")
        return (self._width, self._height)
    
    def generate_tiles(self, tile_size: Optional[int] = None) -> Iterator[Tuple[str, Tuple[int, int, int, int]]]:
        """Generate tile specifications."""
        if self._width is None or self._height is None:
            raise ValueError("Adapter not properly initialized")
            
        tile_size = tile_size or self.tile_size
        
        for row in range(0, self._height, tile_size):
            for col in range(0, self._width, tile_size):
                width = min(tile_size, self._width - col)
                height = min(tile_size, self._height - row)
                
                tile_id = f"{col}_{row}_{width}_{height}"
                bounds = (col, row, width, height)
                
                yield (tile_id, bounds)
    
    def read_tile(self, tile_id: str, bounds: Tuple[int, int, int, int], band: int = 1) -> RasterTile:
        """Read a specific tile."""
        if self._metadata is None:
            raise ValueError("Adapter not properly initialized")
            
        col_off, row_off, width, height = bounds
        window = RasterWindow(col_off, row_off, width, height)
        
        # Use the wrapped loader to read data
        with self._loader.open_lazy(self._file_path) as reader:
            data = reader.read_window(window, band)
        
        # Calculate geographic bounds for this tile
        pixel_size_x, pixel_size_y = self._metadata.pixel_size
        minx = self._metadata.bounds[0] + col_off * pixel_size_x
        maxx = minx + width * pixel_size_x
        maxy = self._metadata.bounds[3] + row_off * pixel_size_y
        miny = maxy + height * pixel_size_y
        
        tile_bounds = (minx, miny, maxx, maxy)
        
        return RasterTile(
            data=data,
            bounds=tile_bounds,
            tile_id=tile_id,
            crs=self._metadata.crs,
            nodata=self._metadata.nodata_value
        )
    
    # Implement abstract methods from Cacheable
    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key."""
        # Simple cache key based on file path and args
        key_parts = [str(self._file_path)]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return "|".join(key_parts)
    
    def estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, RasterTile):
            return obj.data.nbytes
        else:
            # Fallback estimation
            return len(str(obj)) * 2
    
    # Implement additional abstract methods from BaseRasterSource
    def _read_tile_data(self, window: Tuple[int, int, int, int], bands: Optional[List[int]] = None) -> np.ndarray:
        """Read data for a specific window."""
        if self._metadata is None:
            raise ValueError("Metadata not initialized. Call _initialize_source() first.")
            
        col_off, row_off, width, height = window
        raster_window = RasterWindow(col_off, row_off, width, height)
        
        # Use the wrapped loader to read data
        with self._loader.open_lazy(self._file_path) as reader:
            if bands:
                # Read multiple bands
                band_data = []
                for band in bands:
                    data = reader.read_window(raster_window, band)
                    band_data.append(data)
                return np.stack(band_data, axis=0)
            else:
                # Read single band (default: band 1)
                return reader.read_window(raster_window, 1)[np.newaxis, ...]
    
    def _get_window_bounds(self, window: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        """Get geographic bounds for a pixel window."""
        if self._metadata is None:
            raise ValueError("Metadata not initialized. Call _initialize_source() first.")
            
        col_off, row_off, width, height = window
        
        # Calculate geographic bounds for this window
        pixel_size_x, pixel_size_y = self._metadata.pixel_size
        minx = self._metadata.bounds[0] + col_off * pixel_size_x
        maxx = minx + width * pixel_size_x
        maxy = self._metadata.bounds[3] + row_off * pixel_size_y
        miny = maxy + height * pixel_size_y
        
        return (minx, miny, maxx, maxy)
