#!/usr/bin/env python3
"""Create a spatial subset of the real data for testing."""
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds
import numpy as np

# Define small test bounds - 5x5 degree box in Africa (real data region)
test_bounds = [10, 0, 15, 5]  # minx, miny, maxx, maxy

# Process each dataset
datasets = [
    ("/maps/mwd24/richness/daru-plants-richness.tif", "/home/yl998/plants_subset.tif"),
    ("/maps/mwd24/richness/iucn-terrestrial-richness.tif", "/home/yl998/terrestrial_subset.tif")
]

for src_path, dst_path in datasets:
    print(f"Creating subset of {src_path}...")
    
    with rasterio.open(src_path) as src:
        # Get window for subset bounds
        window = from_bounds(*test_bounds, src.transform)
        
        # Read subset data
        subset_data = src.read(1, window=window)
        
        # Calculate new transform
        height, width = subset_data.shape
        subset_transform = transform_from_bounds(*test_bounds, width, height)
        
        # Copy metadata and update
        out_meta = src.meta.copy()
        out_meta.update({
            'height': height,
            'width': width,
            'transform': subset_transform
        })
        
        # Write subset
        with rasterio.open(dst_path, 'w', **out_meta) as dst:
            dst.write(subset_data, 1)
        
        print(f"  Created {dst_path}: {width}x{height} pixels")
        print(f"  Data range: {np.min(subset_data)} to {np.max(subset_data)}")

print(f"\nSubset bounds: {test_bounds}")
print("These are REAL data subsets from the exact same source files!")