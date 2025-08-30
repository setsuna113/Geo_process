from src.abstractions import get_logger
"""
Path loader - validates and prepares file paths for operations.

Following FarmVibes pattern: operations that can work with paths 
(merge, VRT creation) use this loader.
"""

from typing import List, Union, Optional
from pathlib import Path
import logging
import rasterio

logger = get_logger(__name__)


def validate_paths(paths: Union[Path, str, List[Union[Path, str]]]) -> List[Path]:
    """
    Validate raster paths exist and are readable.
    
    This is what operations call when they need paths for rasterio.merge()
    or other path-based operations.
    
    Args:
        paths: Single path or list of paths to validate
        
    Returns:
        List of validated Path objects
        
    Raises:
        FileNotFoundError: If any path doesn't exist
        ValueError: If paths are not valid rasters
    """
    # Ensure list
    if not isinstance(paths, list):
        paths = [paths]
    
    # Convert to Path objects
    path_objects = []
    for p in paths:
        path = Path(p) if not isinstance(p, Path) else p
        
        # Check exists
        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")
        
        # Check is file
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
            
        path_objects.append(path)
    
    # Validate rasters are readable
    for path in path_objects:
        try:
            with rasterio.open(path) as src:
                # Just check it opens
                _ = src.profile
                logger.debug(f"Validated raster: {path.name} ({src.width}x{src.height})")
        except Exception as e:
            raise ValueError(f"Invalid raster {path}: {e}")
    
    logger.info(f"Validated {len(path_objects)} raster paths")
    return path_objects


def get_raster_info(path: Path) -> dict:
    """
    Get raster metadata without loading data.
    
    Args:
        path: Path to raster file
        
    Returns:
        Dictionary with raster information
    """
    with rasterio.open(path) as src:
        info = {
            'path': path,
            'driver': src.driver,
            'dtype': src.dtypes[0],
            'count': src.count,
            'width': src.width,
            'height': src.height,
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'nodata': src.nodata
        }
    return info


def check_alignment(paths: List[Path]) -> bool:
    """
    Check if rasters are aligned (same CRS, resolution, bounds).
    
    Args:
        paths: List of raster paths to check
        
    Returns:
        True if all rasters are aligned
    """
    if len(paths) < 2:
        return True
        
    # Get first raster info as reference
    with rasterio.open(paths[0]) as ref:
        ref_crs = ref.crs
        ref_res = ref.res
        ref_bounds = ref.bounds
    
    # Check others match
    for path in paths[1:]:
        with rasterio.open(path) as src:
            if src.crs != ref_crs:
                logger.warning(f"CRS mismatch: {path.name}")
                return False
            if src.res != ref_res:
                logger.warning(f"Resolution mismatch: {path.name}")
                return False
            # Bounds can be different for merge, just log
            if src.bounds != ref_bounds:
                logger.debug(f"Bounds differ: {path.name}")
    
    return True


def prepare_for_merge(paths: List[Path]) -> dict:
    """
    Prepare paths for merge operation.
    
    Args:
        paths: List of validated paths
        
    Returns:
        Dictionary with merge parameters
    """
    # Get common CRS and resolution
    with rasterio.open(paths[0]) as src:
        out_crs = src.crs
        out_res = src.res
        
    # Calculate output bounds (union of all)
    bounds_list = []
    for path in paths:
        with rasterio.open(path) as src:
            bounds_list.append(src.bounds)
    
    # Get min/max
    west = min(b.left for b in bounds_list)
    south = min(b.bottom for b in bounds_list)
    east = max(b.right for b in bounds_list)
    north = max(b.top for b in bounds_list)
    
    return {
        'paths': paths,
        'crs': out_crs,
        'res': out_res,
        'bounds': (west, south, east, north),
        'aligned': check_alignment(paths)
    }