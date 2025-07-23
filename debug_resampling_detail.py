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

# Initialize everything exactly like the test
config = Config()
config.settings['paths']['data_dir'] = test_data_dir
# Use the same resolution that was causing problems
config.settings.setdefault('resampling', {}).update({
    'target_resolution': 0.2,  # Same as test
    'engine': 'numpy'
})

db = DatabaseManager()
processor = ResamplingProcessor(config, db)

print("=== Configuration ===")
print(f"Target resolution: {processor.target_resolution}")
print(f"Resampling config: {processor.resampling_config}")

# Test the exact dataset config that was failing
dataset_config = {
    'name': 'plants-richness',
    'path_key': 'plants_richness',
    'data_type': 'richness_data',
    'band_name': 'plants_richness'
}

print(f"\n=== Testing dataset: {dataset_config['name']} ===")

try:
    # Step 1: Check file loading
    path_key = dataset_config['path_key']
    data_files = config.get('data_files', {})
    data_dir = Path(config.get('paths.data_dir', 'data'))
    raster_path = data_dir / data_files[path_key]
    
    print(f"Raster path: {raster_path}")
    print(f"File exists: {raster_path.exists()}")
    
    if raster_path.exists():
        with rasterio.open(raster_path) as src:
            print(f"Source shape: {src.shape}")
            print(f"Source bounds: {src.bounds}")
            print(f"Source resolution: {abs(src.transform[0]):.6f}°")
    
    # Step 2: Try the actual resampling
    print(f"\n=== Attempting Resampling ===")
    result = processor.resample_dataset(dataset_config)
    print(f"✅ Success! Result: {result.name}")
    print(f"   Shape: {result.shape}")
    print(f"   Resolution: {result.target_resolution}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    
    # Let's trace the exact failure point
    import traceback
    print("\n=== Full Traceback ===")
    traceback.print_exc()
    
    # Try to isolate where it fails
    print(f"\n=== Manual Step-by-Step Debug ===")
    try:
        # Test catalog loading
        raster_entry = processor.catalog.get_raster(dataset_config['name'])
        print(f"Catalog entry: {raster_entry}")
        
        if raster_entry is None:
            print("Adding to catalog...")
            data_files = processor.config.get('data_files', {})
            print(f"data_files: {data_files}")
            
            if path_key in data_files:
                data_dir = Path(processor.config.get('paths.data_dir', 'data'))
                raster_path = data_dir / data_files[path_key]
                print(f"Will add: {raster_path}")
                
                # Test actual catalog add
                raster_entry = processor.catalog.add_raster(
                    raster_path,
                    dataset_type=dataset_config['data_type'],
                    validate=True
                )
                print(f"Added to catalog: {raster_entry.name}")
        
    except Exception as inner_e:
        print(f"Inner error: {inner_e}")
        traceback.print_exc()