from src.abstractions import get_logger
"""
Memory loader - loads raster data into memory when computation requires it.

Following FarmVibes pattern: only operations that NEED arrays 
(gradient, clip, analysis) use this loader.
"""

from typing import Union, List, Optional, Tuple
from pathlib import Path
import logging
import numpy as np
import xarray as xr
import rasterio
from rasterio.windows import Window

logger = get_logger(__name__)


def load_to_array(
    data_or_path: Union[Path, str, List[Union[Path, str]], xr.DataArray],
    bands: Optional[Union[int, List[int]]] = None,
    window: Optional[Window] = None,
    chunks: Optional[Union[int, Tuple[int, int]]] = None
) -> xr.DataArray:
    """
    Load raster data into memory as xarray DataArray.
    
    This is what operations call when they NEED data in memory for computation
    (e.g., gradient calculation, clipping, statistical analysis).
    
    Args:
        data_or_path: Path(s) to raster or already-loaded DataArray
        bands: Specific band(s) to load (1-indexed for files)
        window: Window to load (for partial loading)
        chunks: Chunk size for dask arrays (memory-efficient processing)
        
    Returns:
        xarray DataArray with loaded data
        
    Note:
        If already a DataArray, returns as-is (no unnecessary copy).
        This follows FarmVibes pattern of loading only when needed.
    """
    # If already loaded, return as-is
    if isinstance(data_or_path, xr.DataArray):
        logger.debug("Data already loaded, returning as-is")
        return data_or_path
    
    # Handle list of paths (merge in memory)
    if isinstance(data_or_path, list):
        logger.info(f"Loading and merging {len(data_or_path)} rasters into memory")
        arrays = []
        for path in data_or_path:
            arr = load_single_raster(path, bands, window, chunks)
            arrays.append(arr)
        # Merge arrays (simplified - real merge would handle alignment)
        return xr.concat(arrays, dim='band') if len(arrays) > 1 else arrays[0]
    
    # Single path
    return load_single_raster(data_or_path, bands, window, chunks)


def load_single_raster(
    path: Union[Path, str],
    bands: Optional[Union[int, List[int]]] = None,
    window: Optional[Window] = None,
    chunks: Optional[Union[int, Tuple[int, int]]] = None
) -> xr.DataArray:
    """
    Load a single raster file into memory.
    
    Args:
        path: Path to raster file
        bands: Band(s) to load (1-indexed)
        window: Window for partial loading
        chunks: Enable dask chunking
        
    Returns:
        xarray DataArray
    """
    path = Path(path) if not isinstance(path, Path) else path
    
    logger.info(f"Loading raster: {path.name}")
    
    # Open with rasterio for controlled loading
    with rasterio.open(path) as src:
        # Determine bands to load
        if bands is not None:
            if isinstance(bands, int):
                bands = [bands]
            band_indexes = bands
        else:
            band_indexes = list(range(1, src.count + 1))
        
        # Read data (with window if specified)
        if window:
            logger.debug(f"Loading window: {window}")
            data = src.read(band_indexes, window=window)
            transform = src.window_transform(window)
        else:
            data = src.read(band_indexes)
            transform = src.transform
        
        # Get coordinates
        if window:
            rows, cols = window.height, window.width
            row_off, col_off = window.row_off, window.col_off
        else:
            rows, cols = src.height, src.width
            row_off, col_off = 0, 0
            
        # Build coordinate arrays
        x_coords = np.arange(col_off, col_off + cols) * transform[0] + transform[2]
        y_coords = np.arange(row_off, row_off + rows) * transform[4] + transform[5]
        
        # Handle single band case
        if len(band_indexes) == 1 and data.shape[0] == 1:
            data = data[0]  # Remove band dimension for single band
            dims = ['y', 'x']
            coords = {'y': y_coords, 'x': x_coords}
        else:
            dims = ['band', 'y', 'x']
            coords = {
                'band': band_indexes,
                'y': y_coords,
                'x': x_coords
            }
        
        # Create DataArray
        arr = xr.DataArray(
            data,
            dims=dims,
            coords=coords,
            attrs={
                'crs': str(src.crs),
                'transform': transform,
                'source': str(path)
            }
        )
        
        # Add rioxarray metadata
        arr.rio.write_crs(src.crs, inplace=True)
        arr.rio.write_transform(transform, inplace=True)
        
        # Apply chunking if requested
        if chunks:
            logger.debug(f"Applying dask chunks: {chunks}")
            if isinstance(chunks, int):
                chunks = {'y': chunks, 'x': chunks}
            arr = arr.chunk(chunks)
        
        # Log memory usage
        memory_mb = data.nbytes / (1024 * 1024)
        logger.info(f"Loaded {path.name}: {data.shape} ({memory_mb:.1f} MB)")
        
        return arr


def load_with_callback(
    path: Path,
    callback,
    bands: Optional[Union[int, List[int]]] = None,
    window: Optional[Window] = None
) -> any:
    """
    Load raster and apply callback function immediately.
    
    Useful for operations that need to process data but don't need
    to keep it in memory.
    
    Args:
        path: Path to raster
        callback: Function to apply to loaded data
        bands: Bands to load
        window: Window to load
        
    Returns:
        Result of callback function
    """
    logger.debug(f"Loading {path.name} with callback")
    
    # Load data
    data = load_single_raster(path, bands, window)
    
    # Apply callback
    result = callback(data)
    
    # Data goes out of scope here (memory freed)
    return result


def estimate_memory_usage(paths: List[Path], window: Optional[Window] = None) -> float:
    """
    Estimate memory usage for loading rasters.
    
    Args:
        paths: List of raster paths
        window: Optional window (reduces memory estimate)
        
    Returns:
        Estimated memory usage in MB
    """
    total_bytes = 0
    
    for path in paths:
        with rasterio.open(path) as src:
            if window:
                shape = (src.count, window.height, window.width)
            else:
                shape = (src.count, src.height, src.width)
            
            # Get dtype size
            dtype = np.dtype(src.dtypes[0])
            bytes_per_pixel = dtype.itemsize
            
            # Calculate size
            total_bytes += np.prod(shape) * bytes_per_pixel
    
    return total_bytes / (1024 * 1024)