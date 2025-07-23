#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, 'src')
sys.modules['pytest'] = type(sys)('pytest')  # Force test mode

import rasterio
import numpy as np
from src.config.config import Config

# Create small test data first if needed
test_data_dir = Path("data/test_richness_small")
test_data_dir.mkdir(exist_ok=True)

source_dir = Path("data/richness_maps")
plants_src = source_dir / "daru-plants-richness.tif"
plants_test = test_data_dir / "daru-plants-richness.tif"

if not plants_test.exists():
    with rasterio.open(plants_src) as src:
        # Get a small window (100x100 pixels from center)
        width, height = src.width, src.height
        center_x, center_y = width // 2, height // 2
        
        window = rasterio.windows.Window(center_x - 50, center_y - 50, 100, 100)
        data = src.read(1, window=window)
        transform = src.window_transform(window)
        
        with rasterio.open(
            plants_test, 'w', driver='GTiff', height=100, width=100,
            count=1, dtype=data.dtype, crs=src.crs, transform=transform, nodata=src.nodata
        ) as dst:
            dst.write(data, 1)

# Now analyze the test file
print("=== Test Dataset Analysis ===")
with rasterio.open(plants_test) as src:
    print(f"Shape: {src.shape}")
    print(f"CRS: {src.crs}")
    print(f"Transform: {src.transform}")
    print(f"Bounds: {src.bounds}")
    
    # Extract values
    minx, miny, maxx, maxy = src.bounds
    print(f"minx: {minx}, miny: {miny}, maxx: {maxx}, maxy: {maxy}")
    print(f"Width (degrees): {maxx - minx}")
    print(f"Height (degrees): {maxy - miny}")
    
    # Test different target resolutions
    target_resolutions = [0.05, 0.1, 0.2, 0.5, 1.0]
    
    print(f"\n=== Target Dimension Calculations ===")
    for target_res in target_resolutions:
        width = int(np.ceil((maxx - minx) / target_res))
        height = int(np.ceil((maxy - miny) / target_res))
        print(f"Target res {target_res}°: width={width}, height={height}")
        
        if width <= 0 or height <= 0:
            print(f"  ❌ NEGATIVE/ZERO DIMENSIONS!")
        else:
            print(f"  ✅ Valid dimensions")

# Check current config
print(f"\n=== Current Config ===")
config = Config()
test_data_override = Path("data/test_richness_small")
config.settings['paths']['data_dir'] = test_data_override

resampling_config = config.get('resampling', {})
current_target_res = resampling_config.get('target_resolution', 0.05)
print(f"Config target resolution: {current_target_res}°")

# Test calculation with current config
width = int(np.ceil((maxx - minx) / current_target_res))
height = int(np.ceil((maxy - miny) / current_target_res))
print(f"Current config would produce: width={width}, height={height}")