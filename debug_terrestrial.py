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

# Create test data (both files)
test_data_dir = Path("data/test_richness_small")
test_data_dir.mkdir(exist_ok=True)

source_dir = Path("data/richness_maps")
plants_src = source_dir / "daru-plants-richness.tif"
terrestrial_src = source_dir / "iucn-terrestrial-richness.tif"

plants_test = test_data_dir / "daru-plants-richness.tif"
terrestrial_test = test_data_dir / "iucn-terrestrial-richness.tif"

for src_file, test_file in [(plants_src, plants_test), (terrestrial_src, terrestrial_test)]:
    if not test_file.exists():
        with rasterio.open(src_file) as src:
            width, height = src.width, src.height
            center_x, center_y = width // 2, height // 2
            
            window = rasterio.windows.Window(center_x - 50, center_y - 50, 100, 100)
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            
            with rasterio.open(
                test_file, 'w', driver='GTiff', height=100, width=100,
                count=1, dtype=data.dtype, crs=src.crs, transform=transform, nodata=src.nodata
            ) as dst:
                dst.write(data, 1)

        print(f"Created {test_file.name}")

# Initialize 
config = Config()
config.settings['paths']['data_dir'] = test_data_dir
config.settings.setdefault('resampling', {}).update({
    'target_resolution': 0.2,
    'engine': 'numpy'
})

db = DatabaseManager()
processor = ResamplingProcessor(config, db)

# Test both datasets
dataset_configs = [
    {
        'name': 'plants-richness',
        'path_key': 'plants_richness',
        'data_type': 'richness_data',
        'band_name': 'plants_richness'
    },
    {
        'name': 'terrestrial-richness',
        'path_key': 'terrestrial_richness', 
        'data_type': 'richness_data',
        'band_name': 'terrestrial_richness'
    }
]

for dataset_config in dataset_configs:
    print(f"\n=== Testing dataset: {dataset_config['name']} ===")
    
    try:
        result = processor.resample_dataset(dataset_config)
        print(f"✅ Success! Result: {result.name}")
        print(f"   Shape: {result.shape}")
        print(f"   Resolution: {result.target_resolution}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        
        # Debug the specific error
        if "Unknown dataset type" in str(e):
            print(f"   Dataset type '{dataset_config['data_type']}' not recognized")
            print(f"   Available strategies: {processor.strategies}")
            
        import traceback
        print("   Traceback:")
        traceback.print_exc()