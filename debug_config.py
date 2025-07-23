#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
sys.modules['pytest'] = type(sys)('pytest')  # Force test mode

from src.config.config import Config
from pathlib import Path

# Initialize config
config = Config()

print("=== Original Config ===")
print(f"data_files: {config.get('data_files', {})}")
print(f"paths.data_dir: {config.get('paths.data_dir')}")

# Test modification like in the test script
test_data_dir = Path("data/test_richness_small")
original_data_dir = config.paths['data_dir']
config.settings['paths']['data_dir'] = test_data_dir

print("\n=== After Modification ===")
print(f"config.paths['data_dir']: {config.paths['data_dir']}")
print(f"config.get('paths.data_dir'): {config.get('paths.data_dir')}")

# Test what the resampling processor would see
data_files = config.get('data_files', {})
path_key = 'plants_richness'
print(f"\ndata_files keys: {list(data_files.keys())}")
print(f"path_key '{path_key}' in data_files: {path_key in data_files}")

if path_key in data_files:
    data_dir = Path(config.get('paths.data_dir', 'data'))
    raster_path = data_dir / data_files[path_key]
    print(f"raster_path: {raster_path}")
    print(f"raster_path exists: {raster_path.exists()}")