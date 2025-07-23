#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, 'src')
sys.modules['pytest'] = type(sys)('pytest')  # Force test mode

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.processors.data_preparation.resampling_processor import ResamplingProcessor

# Create test datasets first
try:
    import rasterio
    from rasterio.windows import from_bounds
    import numpy as np
    
    # Create test data directory
    test_data_dir = Path("data/test_richness_small")
    test_data_dir.mkdir(exist_ok=True)
    
    # Source files
    source_dir = Path("data/richness_maps")
    plants_src = source_dir / "daru-plants-richness.tif"
    terrestrial_src = source_dir / "iucn-terrestrial-richness.tif" 
    
    # Target files (small versions)
    plants_test = test_data_dir / "daru-plants-richness.tif"
    terrestrial_test = test_data_dir / "iucn-terrestrial-richness.tif"
    
    # Create small subset of each dataset
    for src_file, test_file in [(plants_src, plants_test), (terrestrial_src, terrestrial_test)]:
        if not test_file.exists():
            with rasterio.open(src_file) as src:
                # Get a small window (100x100 pixels from center)
                width, height = src.width, src.height
                center_x, center_y = width // 2, height // 2
                
                # Create small window
                window = rasterio.windows.Window(
                    center_x - 50, center_y - 50, 100, 100
                )
                
                # Read data for this window
                data = src.read(1, window=window)
                
                # Get the transform for this window
                transform = src.window_transform(window)
                
                # Write small test file
                with rasterio.open(
                    test_file, 'w',
                    driver='GTiff',
                    height=100, width=100,
                    count=1, dtype=data.dtype,
                    crs=src.crs,
                    transform=transform,
                    nodata=src.nodata
                ) as dst:
                    dst.write(data, 1)
                
                print(f"Created {test_file.name}")
    
except ImportError:
    print("rasterio not available")
    test_data_dir = Path("data/richness_maps")

# Initialize config
config = Config()

# Override with test data directory
original_data_dir = config.paths['data_dir']
config.settings['paths']['data_dir'] = test_data_dir

print(f"Config data_dir: {config.get('paths.data_dir')}")
print(f"data_files: {config.get('data_files', {})}")

# Initialize database and processor
db = DatabaseManager()
processor = ResamplingProcessor(config, db)

print(f"Processor target resolution: {processor.target_resolution}")
print(f"Processor datasets config: {len(processor.datasets_config)} datasets")

# Test dataset configuration
dataset_configs = config.get('datasets.target_datasets', [])
for dataset_config in dataset_configs:
    print(f"\n=== Testing dataset: {dataset_config['name']} ===")
    print(f"path_key: {dataset_config['path_key']}")
    print(f"data_type: {dataset_config['data_type']}")
    
    # Test what processor.resample_dataset would see
    path_key = dataset_config['path_key']
    data_files = config.get('data_files', {})
    print(f"path_key in data_files: {path_key in data_files}")
    
    if path_key in data_files:
        data_dir = Path(config.get('paths.data_dir', 'data'))
        raster_path = data_dir / data_files[path_key]
        print(f"raster_path: {raster_path}")
        print(f"exists: {raster_path.exists()}")
        
        try:
            # Try to resample this dataset with correct signature
            result = processor.resample_dataset(dataset_config)
            print(f"✅ Resampling successful: {result.name}")
        except Exception as e:
            print(f"❌ Resampling failed: {e}")
            import traceback
            traceback.print_exc()