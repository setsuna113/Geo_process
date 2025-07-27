#!/usr/bin/env python3
"""Create EXACT subset of real data by cutting out a small window."""
import rasterio
from rasterio.windows import Window
import numpy as np

# Define a VERY small window - just 100x100 pixels from the original data
# This ensures we use the EXACT same resolution, projection, etc.
# Using regions with confirmed data (from find_data_region.py)
row_offset = 1000   # Start from row 1000 
col_offset = 5000   # Start from column 5000
window_size = 100   # 100x100 pixel window

datasets = [
    ("/maps/mwd24/richness/daru-plants-richness.tif", "/home/yl998/plants_exact_subset.tif"),
    ("/maps/mwd24/richness/iucn-terrestrial-richness.tif", "/home/yl998/terrestrial_exact_subset.tif")
]

for src_path, dst_path in datasets:
    print(f"\nProcessing {src_path}...")
    
    with rasterio.open(src_path) as src:
        # Create window
        window = Window(col_offset, row_offset, window_size, window_size)
        
        # Read the window data
        data = src.read(1, window=window)
        
        # Get the transform for this window
        transform = src.window_transform(window)
        
        # Copy all metadata exactly
        out_meta = src.meta.copy()
        
        # Update only the necessary fields
        out_meta.update({
            'height': window_size,
            'width': window_size,
            'transform': transform
        })
        
        # Write the subset
        with rasterio.open(dst_path, 'w', **out_meta) as dst:
            dst.write(data, 1)
        
        # Verify
        with rasterio.open(dst_path) as verify:
            print(f"  Created: {dst_path}")
            print(f"  Size: {window_size}x{window_size} pixels")
            print(f"  Bounds: {verify.bounds}")
            print(f"  Resolution: {verify.res} (same as original: {src.res})")
            print(f"  CRS: {verify.crs}")
            print(f"  Data type: {verify.dtypes[0]}")
            print(f"  NoData: {verify.nodata}")
            valid_data = data[data != src.nodata]
            if len(valid_data) > 0:
                print(f"  Data range: {np.min(valid_data)} to {np.max(valid_data)}")
            else:
                print(f"  Data range: All nodata values")

print("\nThese subsets are EXACT cuts from the original data!")
print("They preserve all properties: resolution, CRS, data type, nodata, etc.")