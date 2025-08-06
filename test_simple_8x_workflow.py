#!/usr/bin/env python
"""Simple standalone test of 8x workflow bypassing orchestrator"""

import ee
from src.gee_integration.config import GEEConfig
from src.gee_integration.auth import GEEAuthenticator
from src.gee_integration.processing.recursive_resampler import RecursiveResampler
from src.gee_integration.processing import GEEMerger

# Initialize
config = GEEConfig.from_yaml('config/gee_config.yml')
auth = GEEAuthenticator(config.__dict__)
auth.authenticate(
    method=config.auth_method,
    service_account_key=config.service_account_key,
    project_id=config.project_id
)

print("Running simple 8x workflow test...")
print("=" * 60)

# Configuration
resolution_outputs = {'1x': False, '2x': False, '4x': False, '8x': True}
base_resolution = 0.016666666666667
scale_8x = base_resolution * 8 * 111320  # ~14.8km

# Load 2 biodiversity datasets for simplicity
print("1. Loading biodiversity datasets...")
biodiv_images = []
biodiv_configs = []

for name in ['animals', 'plants']:
    asset_id = f'projects/earthengine-legacy/assets/users/geebiodiversity/production/{name}'
    image = ee.Image(asset_id)
    band_name = f'{name[:-1]}_richness'
    image = image.rename([band_name])
    biodiv_images.append(image)
    biodiv_configs.append({
        'name': name,
        'band_name': band_name,
        'data_type': 'continuous',
        'resampling': 'max'
    })
    print(f"   ✓ Loaded {name}")

# Process to 8x
print("\n2. Applying recursive 2x downsampling to 8x...")
resampler = RecursiveResampler(
    base_resolution=base_resolution,
    outputs=resolution_outputs,
    target_crs='EPSG:4326'
)

biodiv_results = resampler.process_collection(biodiv_images, biodiv_configs)
biodiv_8x = biodiv_results.get('8x', [])
print(f"   ✓ Got {len(biodiv_8x)} images at 8x resolution")

# Load and align climate
print("\n3. Loading and aligning climate data...")
merger = GEEMerger(config={'masking': {'enabled': False}})
climate = merger.load_worldclim_v2(['bio01', 'bio12'])
print("   ✓ Loaded WorldClim data")

# Merge everything
print("\n4. Merging biodiversity and climate...")
merged = merger.merge_datasets(
    biodiv_8x,
    climate,
    'addBands',
    mask_ocean=False,
    include_elevation=False,
    scale=scale_8x
)
print("   ✓ Merged successfully")

# Export
print("\n5. Starting export...")
export_region = ee.Geometry.BBox(-180, -60, 180, 75)  # Production bounds

task = ee.batch.Export.image.toCloudStorage(
    image=merged,
    description='simple_8x_workflow_test',
    bucket='biodiv-gee-yl998',
    fileNamePrefix='test_8x/simple_workflow',
    region=export_region,
    scale=scale_8x,
    crs='EPSG:4326',
    fileFormat='GeoTIFF',
    formatOptions={'cloudOptimized': True, 'noData': 0},
    maxPixels=1e13
)

task.start()
print(f"   ✓ Export task started: {task.id}")

# Monitor
import time
print("\n6. Monitoring export...")
for i in range(6):  # Check for 30 seconds
    time.sleep(5)
    task_info = ee.data.getTaskStatus(task.id)[0]
    state = task_info.get('state', 'UNKNOWN')
    error = task_info.get('error_message', '')
    
    print(f"   [{i*5}s] Status: {state}")
    
    if error:
        print(f"   ❌ ERROR: {error}")
        break
    elif state == 'COMPLETED':
        print(f"   ✅ Export completed successfully!")
        break
    elif state == 'FAILED':
        print(f"   ❌ Export failed!")
        break

print("\n" + "=" * 60)
print("Test complete. Check GCS for output if successful.")