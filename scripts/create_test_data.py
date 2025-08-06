#!/usr/bin/env python
"""Create test biodiversity data with ~50% nodata for GEE pipeline testing"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path

def create_test_raster(output_path, width=100, height=100, resolution=0.01, 
                       nodata_percent=0.5, pattern='realistic', 
                       min_val=1, max_val=100, seed=42):
    """
    Create a test raster with specified nodata pattern.
    
    Args:
        output_path: Output file path
        width: Width in pixels
        height: Height in pixels  
        resolution: Resolution in degrees
        nodata_percent: Percentage of nodata pixels
        pattern: Pattern type ('realistic', 'checkerboard', 'random', 'clusters')
        min_val: Minimum data value
        max_val: Maximum data value
        seed: Random seed
    """
    np.random.seed(seed)
    
    # Create base data
    data = np.random.uniform(min_val, max_val, (height, width)).astype(np.float32)
    
    # Create nodata mask based on pattern
    nodata_mask = np.zeros((height, width), dtype=bool)
    
    if pattern == 'realistic':
        # Create realistic patterns with clusters and edges
        # Add some large circular patches
        for _ in range(5):
            cx, cy = np.random.randint(0, width), np.random.randint(0, height)
            radius = np.random.randint(10, 30)
            y, x = np.ogrid[:height, :width]
            mask_circle = (x - cx)**2 + (y - cy)**2 <= radius**2
            nodata_mask |= mask_circle
        
        # Add some irregular edges (simulate coastlines)
        edge_width = 15
        nodata_mask[:edge_width, :] = True  # Top edge
        nodata_mask[-edge_width:, :] = True  # Bottom edge
        
        # Add some scattered small holes
        for _ in range(20):
            x, y = np.random.randint(0, width-5), np.random.randint(0, height-5)
            nodata_mask[y:y+np.random.randint(2,5), x:x+np.random.randint(2,5)] = True
            
    elif pattern == 'checkerboard':
        # Checkerboard pattern
        block_size = 10
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                if (i // block_size + j // block_size) % 2 == 0:
                    nodata_mask[i:i+block_size, j:j+block_size] = True
                    
    elif pattern == 'random':
        # Random pixels
        nodata_mask = np.random.random((height, width)) < nodata_percent
        
    elif pattern == 'clusters':
        # Clustered nodata using Gaussian smoothing
        random_field = np.random.random((height, width))
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(random_field, sigma=5)
        threshold = np.percentile(smoothed, nodata_percent * 100)
        nodata_mask = smoothed < threshold
    
    # Adjust to get closer to target percentage
    current_percent = np.sum(nodata_mask) / (width * height)
    if current_percent < nodata_percent - 0.1:
        # Add more nodata
        n_additional = int((nodata_percent - current_percent) * width * height)
        available_indices = np.where(~nodata_mask)
        selected = np.random.choice(len(available_indices[0]), 
                                  min(n_additional, len(available_indices[0])), 
                                  replace=False)
        nodata_mask[available_indices[0][selected], available_indices[1][selected]] = True
    
    # Apply nodata
    nodata_value = -9999
    data[nodata_mask] = nodata_value
    
    # Create geospatial reference (global extent for testing)
    # Center around 0,0 for easy testing
    west = -resolution * width / 2
    south = -resolution * height / 2
    east = west + resolution * width
    north = south + resolution * height
    
    transform = from_bounds(west, south, east, north, width, height)
    
    # Write raster
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
        nodata=nodata_value,
        compress='lzw'
    ) as dst:
        dst.write(data, 1)
    
    # Calculate statistics
    valid_data = data[data != nodata_value]
    actual_nodata_percent = 1 - (len(valid_data) / (width * height))
    
    print(f"Created {output_path.name}:")
    print(f"  Size: {width}x{height} pixels")
    print(f"  Resolution: {resolution}° ({resolution * 111320:.0f}m at equator)")
    print(f"  NoData: {actual_nodata_percent*100:.1f}%")
    print(f"  Valid range: {valid_data.min():.1f} - {valid_data.max():.1f}")
    print(f"  Extent: [{west:.3f}, {south:.3f}, {east:.3f}, {north:.3f}]")

def main():
    # Create test data directory
    test_dir = Path('/home/jason/geo/data/test_biodiversity')
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating test biodiversity data with ~50% nodata...\n")
    
    # Create different biodiversity layers with varying patterns
    # Animals - realistic pattern with clusters
    create_test_raster(
        test_dir / 'test_animals.tif',
        width=100, height=100,
        resolution=0.01,
        nodata_percent=0.5,
        pattern='realistic',
        min_val=1, max_val=500,
        seed=42
    )
    
    # Plants - different realistic pattern
    create_test_raster(
        test_dir / 'test_plants.tif',
        width=100, height=100,
        resolution=0.01,
        nodata_percent=0.45,  # Slightly different coverage
        pattern='realistic',
        min_val=10, max_val=1000,
        seed=43
    )
    
    # Fungi - clustered pattern
    create_test_raster(
        test_dir / 'test_fungi.tif',
        width=100, height=100,
        resolution=0.01,
        nodata_percent=0.55,  # More sparse
        pattern='clusters',
        min_val=5, max_val=200,
        seed=44
    )
    
    print(f"\n✅ Test data created in {test_dir}")
    print("\nTest data characteristics:")
    print("- 100x100 pixels each (1° x 1° at 0.01° resolution)")
    print("- ~50% nodata with realistic patterns")
    print("- Different nodata patterns for each biodiversity type")
    print("- Centered around 0°,0° for easy verification")

if __name__ == '__main__':
    main()