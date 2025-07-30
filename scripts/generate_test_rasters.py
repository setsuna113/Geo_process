#!/usr/bin/env python3
"""Generate test raster files for pipeline testing."""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_test_raster(filename, width=100, height=100, bounds=(-180, -90, 180, 90), 
                      pattern='gradient', seed=None):
    """Create a test raster file with specified pattern."""
    
    if seed is not None:
        np.random.seed(seed)
    
    # Create data based on pattern
    if pattern == 'gradient':
        # Create gradient from NW to SE
        data = np.zeros((height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                data[i, j] = (i + j) / (height + width) * 100
    elif pattern == 'random':
        # Random values between 0 and 100
        data = np.random.rand(height, width).astype(np.float32) * 100
    elif pattern == 'checkerboard':
        # Checkerboard pattern
        data = np.zeros((height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                if (i // 10 + j // 10) % 2 == 0:
                    data[i, j] = 75
                else:
                    data[i, j] = 25
    elif pattern == 'hotspots':
        # Random hotspots
        data = np.ones((height, width), dtype=np.float32) * 10
        # Add 5 hotspots
        for _ in range(5):
            y, x = np.random.randint(10, height-10), np.random.randint(10, width-10)
            radius = np.random.randint(5, 15)
            for i in range(max(0, y-radius), min(height, y+radius)):
                for j in range(max(0, x-radius), min(width, x+radius)):
                    dist = np.sqrt((i-y)**2 + (j-x)**2)
                    if dist < radius:
                        data[i, j] = max(data[i, j], 90 * (1 - dist/radius))
    
    # Create transform
    transform = from_bounds(*bounds, width, height)
    
    # Write raster
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.float32,
        crs='EPSG:4326',
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(data, 1)
        # Add some metadata
        dst.update_tags(
            pattern=pattern,
            test_data=True,
            description=f'Test raster with {pattern} pattern'
        )
    
    print(f"✅ Created {filename} ({pattern} pattern, {width}x{height})")

def main():
    """Generate test raster files."""
    # Create output directory
    output_dir = Path('/home/yl998/dev/geo/data/test_rasters')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Generate 4 test rasters with different patterns and resolutions
    # Using resolutions closer to our target (0.5°) to avoid extreme upsampling
    test_files = [
        ('climate_temperature.tiff', 'gradient', 42, 200, 100, 0.9),    # 0.9° resolution
        ('climate_precipitation.tiff', 'random', 123, 300, 150, 0.6),   # 0.6° resolution
        ('landcover_forest.tiff', 'checkerboard', 456, 180, 90, 1.0),   # 1.0° resolution
        ('species_richness.tiff', 'hotspots', 789, 360, 180, 0.5)       # 0.5° resolution (matches target)
    ]
    
    for filename, pattern, seed, width, height, resolution in test_files:
        filepath = output_dir / filename
        # Calculate bounds to maintain square pixels
        lon_extent = width * resolution
        lat_extent = height * resolution
        bounds = (-lon_extent/2, -lat_extent/2, lon_extent/2, lat_extent/2)
        
        create_test_raster(
            filepath,
            width=width,
            height=height,
            bounds=bounds,
            pattern=pattern,
            seed=seed
        )
        print(f"  Resolution: {resolution}° per pixel")
    
    print("\n✅ All test rasters created successfully!")
    
    # Verify files
    print("\n📊 File information:")
    for filename, _, _, _, _, _ in test_files:
        filepath = output_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            with rasterio.open(filepath) as src:
                print(f"  - {filename}: {size_kb:.1f} KB, {src.width}x{src.height}, CRS: {src.crs}")

if __name__ == '__main__':
    main()