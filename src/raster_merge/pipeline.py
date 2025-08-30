from src.abstractions import get_logger
"""
Actual Unix-style pipeline for raster operations.

No state. No magic. Just functions that compose.
"""

from typing import List, Optional, Callable, Any
from pathlib import Path
import logging
from contextlib import ExitStack

from .config import RasterConfig
from .merger_raster.vrt_merger import VRTMerger

logger = get_logger(__name__)


def merge_rasters(input_paths: List[Path], 
                  output_path: Path,
                  config: Optional[RasterConfig] = None) -> Path:
    """
    Merge multiple rasters into one.
    
    Simple function: paths in, path out. No state.
    
    Args:
        input_paths: List of input raster paths
        output_path: Where to write the merged raster
        config: Optional configuration
    
    Returns:
        Path to merged raster
    """
    if not input_paths:
        raise ValueError("No input paths provided")
    
    if len(input_paths) == 1:
        # Single input - just return it
        logger.info("Single input provided, no merge needed")
        return input_paths[0]
    
    # Create config if not provided
    if config is None:
        config = RasterConfig(
            input_paths=input_paths,
            output_dir=output_path.parent,
            output_name=output_path.stem
        )
    else:
        # Update config with our paths
        config.input_paths = input_paths
        config.output_dir = output_path.parent
        config.output_name = output_path.stem
    
    # Use VRTMerger
    merger = VRTMerger(config)
    return merger.merge()


def clip_raster(input_path: Path,
                output_path: Path,
                geometry: Any,
                all_touched: bool = False) -> Path:
    """
    Clip a raster to a geometry.
    
    Args:
        input_path: Input raster path
        output_path: Output raster path
        geometry: Clipping geometry (shapely, GeoJSON, WKT)
        all_touched: Include all pixels touched by geometry
    
    Returns:
        Path to clipped raster
    """
    import rasterio
    from rasterio.mask import mask
    import geopandas as gpd
    from shapely.geometry import mapping
    
    # Convert geometry to GeoJSON format if needed
    if hasattr(geometry, 'geometry'):
        # It's a GeoDataFrame or GeoSeries
        geom_json = [mapping(geom) for geom in geometry.geometry]
    elif hasattr(geometry, '__geo_interface__'):
        # It's a shapely geometry
        geom_json = [mapping(geometry)]
    else:
        # Assume it's already GeoJSON
        geom_json = [geometry]
    
    with rasterio.open(input_path) as src:
        # Perform the clip
        out_image, out_transform = mask(
            src, 
            geom_json, 
            crop=True,
            all_touched=all_touched
        )
        
        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "zstd"
        })
        
        # Write output
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
    
    logger.info(f"Clipped raster written to {output_path}")
    return output_path


def compute_gradient(input_path: Path,
                    output_path: Path,
                    band: int = 1) -> Path:
    """
    Compute gradient magnitude for a raster.
    
    Args:
        input_path: Input raster path
        output_path: Output gradient path
        band: Which band to process
    
    Returns:
        Path to gradient raster
    """
    import rasterio
    import numpy as np
    from scipy.ndimage import sobel
    
    with rasterio.open(input_path) as src:
        # Read the band
        data = src.read(band)
        
        # Compute gradients
        dx = sobel(data, axis=1)
        dy = sobel(data, axis=0)
        gradient = np.sqrt(dx**2 + dy**2)
        
        # Update metadata for output
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "count": 1,
            "dtype": "float32",
            "compress": "zstd"
        })
        
        # Write gradient
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(gradient.astype(np.float32), 1)
    
    logger.info(f"Gradient written to {output_path}")
    return output_path


def build_overviews(raster_path: Path, 
                   levels: Optional[List[int]] = None) -> Path:
    """
    Build pyramid overviews for a raster.
    
    Modifies the raster in-place.
    
    Args:
        raster_path: Path to raster
        levels: Overview levels (default: [2, 4, 8, 16])
    
    Returns:
        Same path (modified in place)
    """
    import rasterio
    from rasterio.enums import Resampling
    
    if levels is None:
        levels = [2, 4, 8, 16]
    
    with rasterio.open(raster_path, 'r+') as dst:
        dst.build_overviews(levels, Resampling.average)
        dst.update_tags(ns='rio_overview', resampling='average')
    
    logger.info(f"Built overviews for {raster_path}")
    return raster_path


# Unix-style pipe operator
def pipe(*operations: Callable) -> Callable:
    """
    Compose operations into a pipeline.
    
    True Unix style - each operation is a function that
    takes input and produces output. No state.
    
    Example:
        pipeline = pipe(
            lambda paths: merge_rasters(paths, Path("merged.tif")),
            lambda path: clip_raster(path, Path("clipped.tif"), geometry),
            lambda path: compute_gradient(path, Path("gradient.tif"))
        )
        result = pipeline(input_paths)
    
    Args:
        *operations: Functions to compose
    
    Returns:
        Composed function
    """
    def pipeline(data):
        result = data
        for op in operations:
            result = op(result)
        return result
    return pipeline


# Convenience function for common use case
def merge_and_export(input_paths: List[Path],
                    output_path: Path,
                    config: Optional[RasterConfig] = None,
                    build_overview: bool = True) -> Path:
    """
    The most common operation - merge rasters and export.
    
    This is what 90% of users actually need.
    
    Args:
        input_paths: Input raster paths
        output_path: Output path
        config: Optional configuration
        build_overview: Whether to build overviews
    
    Returns:
        Path to merged raster
    """
    # Merge
    merged = merge_rasters(input_paths, output_path, config)
    
    # Build overviews if requested
    if build_overview:
        build_overviews(merged)
    
    return merged