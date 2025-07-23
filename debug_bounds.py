#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, 'src')
sys.modules['pytest'] = type(sys)('pytest')  # Force test mode

import rasterio
import numpy as np
from src.config.config import Config
from src.database.connection import DatabaseManager
from src.processors.data_preparation.resampling_processor import ResamplingProcessor

# Create test data
test_data_dir = Path("data/test_richness_small")
test_data_dir.mkdir(exist_ok=True)

source_dir = Path("data/richness_maps")
plants_src = source_dir / "daru-plants-richness.tif"
plants_test = test_data_dir / "daru-plants-richness.tif"

if not plants_test.exists():
    with rasterio.open(plants_src) as src:
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

# Initialize 
config = Config()
config.settings['paths']['data_dir'] = test_data_dir
config.settings.setdefault('resampling', {}).update({
    'target_resolution': 0.2,
    'engine': 'numpy'
})

db = DatabaseManager()
processor = ResamplingProcessor(config, db)

# Check file bounds directly
print("=== Direct File Bounds ===")
with rasterio.open(plants_test) as src:
    direct_bounds = src.bounds
    print(f"Direct bounds: {direct_bounds}")
    minx, miny, maxx, maxy = direct_bounds
    print(f"Width: {maxx - minx:.6f}°")  
    print(f"Height: {maxy - miny:.6f}°")

# Add to catalog and check catalog bounds
print(f"\n=== Catalog Bounds ===")
raster_entry = processor.catalog.add_raster(
    plants_test,
    dataset_type='richness_data', 
    validate=True
)
catalog_bounds = raster_entry.bounds
print(f"Catalog bounds: {catalog_bounds}")
print(f"Bounds type: {type(catalog_bounds)}")

if hasattr(catalog_bounds, '__len__') and len(catalog_bounds) == 4:
    c_minx, c_miny, c_maxx, c_maxy = catalog_bounds
    print(f"Catalog width: {c_maxx - c_minx:.6f}°")
    print(f"Catalog height: {c_maxy - c_miny:.6f}°")
    
    # Test the target shape calculation with catalog bounds
    target_res = 0.2
    width = int(np.ceil((c_maxx - c_minx) / target_res))
    height = int(np.ceil((c_maxy - c_miny) / target_res))
    print(f"Target shape from catalog bounds: ({height}, {width})")
    
    if width <= 0 or height <= 0:
        print("❌ NEGATIVE DIMENSIONS FROM CATALOG BOUNDS!")
        print(f"c_minx={c_minx}, c_miny={c_miny}, c_maxx={c_maxx}, c_maxy={c_maxy}")
        
        # Check if bounds are swapped
        if c_minx > c_maxx:
            print("❌ minx > maxx - bounds are swapped!")
        if c_miny > c_maxy:
            print("❌ miny > maxy - bounds are swapped!")
    else:
        print("✅ Valid dimensions from catalog bounds")

# Test the resampler engine creation
print(f"\n=== Resampler Engine ===")
try:
    method = processor.strategies.get('richness_data', 'sum')
    print(f"Method for richness_data: {method}")
    resampler = processor._create_resampler_engine(method, catalog_bounds)
    print(f"Resampler created: {type(resampler)}")
    print(f"Resampler config: {resampler.config}")
    
    # Test the calculate_output_shape method directly
    target_shape = resampler.calculate_output_shape(catalog_bounds)
    print(f"calculate_output_shape result: {target_shape}")
    
except Exception as e:
    print(f"❌ Resampler creation failed: {e}")
    import traceback
    traceback.print_exc()