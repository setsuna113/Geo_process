#!/usr/bin/env python3
"""
Example of the clean raster merge API.

Simple functions. No magic. No state.
"""

from pathlib import Path
from raster_merge import (
    merge_rasters,
    clip_raster,
    compute_gradient,
    build_overviews,
    pipe,
    RasterConfig
)
from shapely.geometry import box


def example_simple_merge():
    """Simplest possible merge."""
    input_paths = [
        Path("raster1.tif"),
        Path("raster2.tif"),
        Path("raster3.tif")
    ]
    
    # One line to merge
    output = merge_rasters(input_paths, Path("output/merged.tif"))
    print(f"Merged to: {output}")


def example_merge_with_options():
    """Merge with configuration."""
    input_paths = [Path(f"data/raster_{i}.tif") for i in range(10)]
    
    config = RasterConfig(
        input_paths=input_paths,
        output_dir=Path("output"),
        output_name="merged_mean",
        overlap_method="mean",  # Average overlapping pixels
        resolution_strategy="highest",  # Use finest resolution
        compress="zstd",
        compression_level=9
    )
    
    output = merge_rasters(input_paths, Path("output/merged_mean.tif"), config)
    print(f"Merged with config: {output}")


def example_clip():
    """Clip a raster to an area of interest."""
    input_raster = Path("data/large_raster.tif")
    
    # Define area of interest (lon_min, lat_min, lon_max, lat_max)
    aoi = box(-122.5, 37.7, -122.3, 37.9)  # San Francisco area
    
    output = clip_raster(input_raster, Path("output/sf_area.tif"), aoi)
    print(f"Clipped to: {output}")


def example_gradient():
    """Compute gradient (slope) of elevation data."""
    dem_path = Path("data/elevation.tif")
    
    gradient = compute_gradient(dem_path, Path("output/slope.tif"))
    print(f"Gradient computed: {gradient}")


def example_pipeline():
    """Compose operations Unix-style."""
    input_paths = [Path(f"data/tile_{i}.tif") for i in range(5)]
    aoi = box(-122.5, 37.7, -122.3, 37.9)
    
    # Build a pipeline
    pipeline = pipe(
        # Step 1: Merge tiles
        lambda paths: merge_rasters(paths, Path("temp/merged.tif")),
        
        # Step 2: Clip to area of interest
        lambda merged: clip_raster(merged, Path("temp/clipped.tif"), aoi),
        
        # Step 3: Compute gradient
        lambda clipped: compute_gradient(clipped, Path("temp/gradient.tif")),
        
        # Step 4: Build overviews for fast rendering
        lambda gradient: build_overviews(gradient)
    )
    
    # Execute pipeline
    result = pipeline(input_paths)
    print(f"Pipeline result: {result}")


def example_large_scale():
    """Handle hundreds of rasters efficiently."""
    # This works with 1000+ files thanks to automatic batching
    input_paths = [Path(f"tiles/tile_{x}_{y}.tif") 
                   for x in range(50) 
                   for y in range(20)]  # 1000 tiles
    
    # The merger automatically batches to avoid "too many open files"
    output = merge_rasters(input_paths, Path("output/huge_mosaic.tif"))
    print(f"Merged {len(input_paths)} tiles to: {output}")


if __name__ == "__main__":
    print("Raster Merge Examples")
    print("=" * 50)
    
    # Run examples (comment out if files don't exist)
    # example_simple_merge()
    # example_merge_with_options()
    # example_clip()
    # example_gradient()
    # example_pipeline()
    # example_large_scale()
    
    print("\nSee the function implementations for usage patterns")