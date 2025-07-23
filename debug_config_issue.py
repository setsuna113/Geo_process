#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, 'src')
sys.modules['pytest'] = type(sys)('pytest')  # Force test mode

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.processors.data_preparation.resampling_processor import ResamplingProcessor

# Initialize config
config = Config()

# Override with test data directory
test_data_dir = Path("data/test_richness_small")
original_data_dir = config.paths['data_dir']
config.settings['paths']['data_dir'] = test_data_dir

print("=== Config State ===")
print(f"Config object id: {id(config)}")
print(f"Config data_files: {config.get('data_files', {})}")
print(f"Config paths.data_dir: {config.get('paths.data_dir')}")

# Initialize database and processor
db = DatabaseManager()
processor = ResamplingProcessor(config, db)

print(f"\n=== Processor State ===")
print(f"Processor config object id: {id(processor.config)}")
print(f"Processor config data_files: {processor.config.get('data_files', {})}")
print(f"Processor config paths.data_dir: {processor.config.get('paths.data_dir')}")

# Check catalog config
print(f"\n=== Catalog State ===")
print(f"Catalog config object id: {id(processor.catalog.config)}")
print(f"Catalog config data_files: {processor.catalog.config.get('data_files', {})}")
print(f"Catalog config paths.data_dir: {processor.catalog.config.get('paths.data_dir')}")

# Test the exact check that's failing
dataset_config = {
    'name': 'plants-richness',
    'path_key': 'plants_richness',
    'data_type': 'richness_data',
    'band_name': 'plants_richness'
}

print(f"\n=== Direct Check ===")
path_key = dataset_config['path_key']  
data_files = processor.config.get('data_files', {})
print(f"path_key: {path_key}")
print(f"data_files from processor.config: {data_files}")
print(f"path_key in data_files: {path_key in data_files}")

# Also test the catalog check
try:
    raster_entry = processor.catalog.get_raster(dataset_config['name'])
    print(f"Raster entry from catalog: {raster_entry}")
except Exception as e:
    print(f"Catalog get_raster failed: {e}")