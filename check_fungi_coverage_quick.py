#!/usr/bin/env python
"""Quick coverage check using sampling."""

import rasterio
import numpy as np

# Quick coverage check using sampling
am_path = "/maps/spun/AM_fungi/AM_Fungi_Richness_Predicted.tif"
ecm_path = "/maps/spun/EcM_fungi/EcM_Fungi_Richness_Predicted.tif"

print("QUICK FUNGI COVERAGE CHECK (using 1% sample)")
print("=" * 50)

for path, name in [(am_path, "AM Fungi"), (ecm_path, "EcM Fungi")]:
    with rasterio.open(path) as src:
        # Read metadata
        print(f"\n{name}:")
        print(f"  Dimensions: {src.width} x {src.height}")
        print(f"  NoData value: {src.nodata}")
        
        # Sample random windows instead of reading full raster
        n_samples = 100
        window_size = 200
        
        valid_count = 0
        total_count = 0
        values = []
        
        np.random.seed(42)
        for _ in range(n_samples):
            # Random window
            row = np.random.randint(0, src.height - window_size)
            col = np.random.randint(0, src.width - window_size)
            window = rasterio.windows.Window(col, row, window_size, window_size)
            
            # Read window
            data = src.read(1, window=window)
            
            # Count valid pixels
            valid_mask = (data != src.nodata) & np.isfinite(data)
            valid_count += np.sum(valid_mask)
            total_count += data.size
            
            # Collect some valid values
            if np.any(valid_mask):
                values.extend(data[valid_mask].tolist()[:100])
        
        # Estimate coverage
        coverage = valid_count / total_count * 100
        print(f"  Estimated coverage: {coverage:.1f}%")
        
        if values:
            values = np.array(values[:10000])  # Limit to 10k values
            print(f"  Value range: {values.min():.1f} - {values.max():.1f}")
            print(f"  Mean: {values.mean():.1f}")
            print(f"  Median: {np.median(values):.1f}")

# Check geographic extent
with rasterio.open(am_path) as src:
    bounds = src.bounds
    print(f"\nGeographic extent:")
    print(f"  West: {bounds.left:.2f}")
    print(f"  East: {bounds.right:.2f}")
    print(f"  South: {bounds.bottom:.2f}")
    print(f"  North: {bounds.top:.2f}")
    print(f"  Resolution: ~1km ({src.res[0]:.6f} degrees)")
    
# Check what regions have data by sampling lat bands
print("\nCoverage by latitude band (sampling):")
with rasterio.open(am_path) as am_src, rasterio.open(ecm_path) as ecm_src:
    lat_bands = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]
    
    for lat_min, lat_max in lat_bands:
        # Convert lat to row indices (approximate)
        row_max = int((90 - lat_min) / 180 * am_src.height)
        row_min = int((90 - lat_max) / 180 * am_src.height)
        
        # Sample middle of band
        row_sample = (row_min + row_max) // 2
        
        # Read a strip across the longitude
        am_strip = am_src.read(1, window=rasterio.windows.Window(0, row_sample, am_src.width, 10))
        ecm_strip = ecm_src.read(1, window=rasterio.windows.Window(0, row_sample, ecm_src.width, 10))
        
        am_valid = np.mean((am_strip != am_src.nodata) & np.isfinite(am_strip)) * 100
        ecm_valid = np.mean((ecm_strip != ecm_src.nodata) & np.isfinite(ecm_strip)) * 100
        
        print(f"  {lat_min:3d}° to {lat_max:3d}°: AM={am_valid:4.1f}%, EcM={ecm_valid:4.1f}%")