#!/usr/bin/env python3
"""Debug script to test where passthrough is hanging"""

import sys
import time
sys.path.insert(0, '.')

# Test rasterio loading
print("Testing rasterio loading...")
import rasterio
import numpy as np

raster_path = "/maps/mwd24/richness/iucn-terrestrial-richness.tif"  # Smaller file

print(f"Opening {raster_path}...")
start = time.time()

try:
    with rasterio.open(raster_path) as src:
        print(f"  File opened in {time.time() - start:.1f}s")
        print(f"  Shape: {src.height} x {src.width}")
        print(f"  Data type: {src.dtypes[0]}")
        print(f"  Memory estimate: {src.height * src.width * 4 / 1024 / 1024:.1f}MB")
        
        print("  Reading data...")
        read_start = time.time()
        data = src.read(1).astype(np.float32)
        read_time = time.time() - read_start
        print(f"  Data read in {read_time:.1f}s")
        print(f"  Actual shape: {data.shape}")
        print(f"  Actual memory: {data.nbytes / 1024 / 1024:.1f}MB")
        
        # Check for nodata
        if src.nodata is not None:
            print(f"  Handling nodata value: {src.nodata}")
            data[data == src.nodata] = np.nan
            
        print(f"  Non-NaN values: {np.sum(~np.isnan(data)):,}")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print(f"\nTotal time: {time.time() - start:.1f}s")