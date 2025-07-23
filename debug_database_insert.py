#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, 'src')
sys.modules['pytest'] = type(sys)('pytest')  # Force test mode

import rasterio
import numpy as np
import json
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

# Manually run through the resampling process to check the data being inserted
dataset_config = {
    'name': 'plants-richness',
    'path_key': 'plants_richness',
    'data_type': 'richness_data',
    'band_name': 'plants_richness'
}

print("=== Debugging Database Insert Issue ===")

try:
    # Load and resample (but catch the database error)
    dataset_name = dataset_config['name']
    path_key = dataset_config['path_key']
    data_type = dataset_config['data_type']
    band_name = dataset_config['band_name']

    # Add to catalog
    data_files = processor.config.get('data_files', {})
    data_dir = Path(processor.config.get('paths.data_dir', 'data'))
    raster_path = data_dir / data_files[path_key]
    
    raster_entry = processor.catalog.add_raster(
        raster_path,
        dataset_type=data_type,
        validate=True
    )
    
    # Load source data
    source_data = raster_entry.metadata.get('source_data')
    if source_data is None:
        with rasterio.open(raster_path) as src:
            source_data = src.read(1)
    
    source_bounds = raster_entry.bounds
    method = processor.strategies.get(data_type, 'sum')
    
    # Create resampler and resample
    resampler = processor._create_resampler_engine(method, source_bounds)
    result = resampler.resample(
        source_data=source_data,
        source_bounds=source_bounds
    )
    
    # Create resampled dataset info - THIS is where the problematic data is created
    from src.processors.data_preparation.resampling_processor import ResampledDatasetInfo
    resampled_info = ResampledDatasetInfo(
        name=dataset_name,
        source_path=raster_path,
        target_resolution=processor.target_resolution,
        target_crs=processor.target_crs,
        bounds=result.bounds,
        shape=result.data.shape,
        data_type=data_type,
        resampling_method=method,
        band_name=band_name,
        metadata=result.metadata or {}
    )
    
    print("=== Checking ResampledDatasetInfo fields ===")
    print(f"name: {resampled_info.name} ({type(resampled_info.name)})")
    print(f"source_path: {resampled_info.source_path} ({type(resampled_info.source_path)})")
    print(f"target_resolution: {resampled_info.target_resolution} ({type(resampled_info.target_resolution)})")
    print(f"target_crs: {resampled_info.target_crs} ({type(resampled_info.target_crs)})")
    print(f"bounds: {resampled_info.bounds} ({type(resampled_info.bounds)})")
    print(f"shape: {resampled_info.shape} ({type(resampled_info.shape)})")
    print(f"data_type: {resampled_info.data_type} ({type(resampled_info.data_type)})")
    print(f"resampling_method: {resampled_info.resampling_method} ({type(resampled_info.resampling_method)})")
    print(f"band_name: {resampled_info.band_name} ({type(resampled_info.band_name)})")
    print(f"metadata: {resampled_info.metadata} ({type(resampled_info.metadata)})")
    
    # Check if metadata can be JSON serialized
    try:
        json_metadata = json.dumps(resampled_info.metadata)
        print(f"✅ Metadata is JSON serializable")
    except Exception as json_e:
        print(f"❌ Metadata is NOT JSON serializable: {json_e}")
        
        # Find the problematic key
        if isinstance(resampled_info.metadata, dict):
            for key, value in resampled_info.metadata.items():
                try:
                    json.dumps(value)
                except Exception as inner_e:
                    print(f"  ❌ Key '{key}' not serializable: {value} ({type(value)}) - {inner_e}")

    # Test the exact database insert parameters
    print(f"\n=== Database Insert Parameters ===")
    table_name = f"resampled_{resampled_info.name.replace('-', '_')}"
    params = (
        resampled_info.name,
        str(resampled_info.source_path),
        resampled_info.target_resolution,
        resampled_info.target_crs,
        list(resampled_info.bounds),  # Convert to list for JSON storage
        resampled_info.shape[0],
        resampled_info.shape[1],
        resampled_info.data_type,
        resampled_info.resampling_method,
        resampled_info.band_name,
        table_name,
        resampled_info.metadata
    )
    
    for i, param in enumerate(params):
        print(f"  Param {i}: {param} ({type(param)})")
        if isinstance(param, dict):
            print(f"    ❌ DICT FOUND AT PARAM {i}")
            
except Exception as e:
    print(f"❌ Error during resampling: {e}")
    import traceback
    traceback.print_exc()