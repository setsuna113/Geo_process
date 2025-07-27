#!/usr/bin/env python3
"""Find a region with actual data in the rasters."""
import rasterio
import numpy as np

# Check both files
for file_path in ['/maps/mwd24/richness/daru-plants-richness.tif', 
                  '/maps/mwd24/richness/iucn-terrestrial-richness.tif']:
    print(f"\nChecking {file_path}...")
    
    with rasterio.open(file_path) as src:
        print(f"Total size: {src.width}x{src.height}")
        
        # Try different regions
        found = False
        for row_offset in [1000, 2000, 3000, 4000, 5000, 6000]:
            for col_offset in [5000, 10000, 15000, 20000]:
                if col_offset + 100 > src.width or row_offset + 100 > src.height:
                    continue
                    
                data = src.read(1, window=rasterio.windows.Window(col_offset, row_offset, 100, 100))
                valid_count = np.sum(data != src.nodata)
                
                if valid_count > 5000:  # At least 50% valid data
                    valid_data = data[data != src.nodata]
                    print(f"  Found good region at row={row_offset}, col={col_offset}")
                    print(f"  Valid pixels: {valid_count}/10000")
                    print(f"  Data range: {np.min(valid_data)} to {np.max(valid_data)}")
                    
                    # Get bounds for this window
                    window = rasterio.windows.Window(col_offset, row_offset, 100, 100)
                    bounds = rasterio.windows.bounds(window, src.transform)
                    print(f"  Geographic bounds: {bounds}")
                    found = True
                    break
            if found:
                break
                
        if not found:
            print("  No good data region found in tested areas")