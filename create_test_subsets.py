#!/usr/bin/env python3
"""Create EXACT subset of real data by cutting out a small window.
This preserves all properties: resolution, CRS, data type, nodata values, etc.
"""
import rasterio
from rasterio.windows import Window
import numpy as np
import os

# Define window parameters - 500x500 pixels from a region with data
row_offset = 1000   # Start from row 1000 
col_offset = 5000   # Start from column 5000
window_size = 500   # 500x500 pixel window

# Output directory in user's home
output_dir = "/home/yl998/test_data"
os.makedirs(output_dir, exist_ok=True)

datasets = [
    ("/maps/mwd24/richness/daru-plants-richness.tif", 
     os.path.join(output_dir, "plants_test_subset.tif")),
    ("/maps/mwd24/richness/iucn-terrestrial-richness.tif", 
     os.path.join(output_dir, "terrestrial_test_subset.tif"))
]

for src_path, dst_path in datasets:
    print(f"\nProcessing {src_path}...")
    
    with rasterio.open(src_path) as src:
        print(f"  Original size: {src.width}x{src.height}")
        print(f"  Original resolution: {src.res}")
        print(f"  Original CRS: {src.crs}")
        print(f"  Original dtype: {src.dtypes[0]}")
        print(f"  Original nodata: {src.nodata}")
        
        # Create window
        window = Window(col_offset, row_offset, window_size, window_size)
        
        # Read the window data
        data = src.read(1, window=window)
        
        # Get the transform for this window
        transform = src.window_transform(window)
        
        # Copy ALL metadata exactly
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
            print(f"\n  Created: {dst_path}")
            print(f"  Size: {window_size}x{window_size} pixels")
            print(f"  Bounds: {verify.bounds}")
            print(f"  Resolution: {verify.res} (matches original: {src.res})")
            print(f"  CRS: {verify.crs} (matches original: {src.crs})")
            print(f"  Data type: {verify.dtypes[0]} (matches original: {src.dtypes[0]})")
            print(f"  NoData: {verify.nodata} (matches original: {src.nodata})")
            
            valid_data = data[data != src.nodata]
            if len(valid_data) > 0:
                print(f"  Valid pixels: {len(valid_data)}/{window_size*window_size}")
                print(f"  Data range: {np.min(valid_data)} to {np.max(valid_data)}")
                print(f"  Mean value: {np.mean(valid_data):.2f}")
            else:
                print(f"  WARNING: All nodata values in this window!")

print("\n✅ Test subsets created successfully!")
print("These are EXACT cuts from the original data with all properties preserved.")