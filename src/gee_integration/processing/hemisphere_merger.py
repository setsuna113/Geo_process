"""
Merge hemisphere exports into a single GeoTIFF
"""

import rasterio
from rasterio.merge import merge
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


def merge_hemispheres(west_path: Path, east_path: Path, output_path: Path) -> Path:
    """
    Merge western and eastern hemisphere GeoTIFFs into a single global file.
    
    Args:
        west_path: Path to western hemisphere GeoTIFF
        east_path: Path to eastern hemisphere GeoTIFF
        output_path: Path for merged output
        
    Returns:
        Path to merged GeoTIFF
    """
    logger.info(f"Merging hemispheres: {west_path.name} + {east_path.name}")
    
    # Open both files
    datasets = []
    try:
        west_ds = rasterio.open(west_path)
        east_ds = rasterio.open(east_path)
        datasets = [west_ds, east_ds]
        
        # Merge
        merged_data, merged_transform = merge(datasets)
        
        # Get metadata from first file
        out_meta = west_ds.meta.copy()
        
        # Update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": merged_data.shape[1],
            "width": merged_data.shape[2],
            "transform": merged_transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512
        })
        
        # Write merged file
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(merged_data)
        
        logger.info(f"✅ Merged hemispheres to: {output_path}")
        return output_path
        
    finally:
        # Close datasets
        for ds in datasets:
            if hasattr(ds, 'close'):
                ds.close()