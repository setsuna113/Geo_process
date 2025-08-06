#!/usr/bin/env python
"""Debug the full processing chain to find where projection is lost"""

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

print("Debugging full processing chain...")
print("=" * 60)

# Setup
resolution_outputs = {'1x': False, '2x': False, '4x': False, '8x': True}
base_resolution = 0.016666666666667

# Step 1: Load biodiversity images
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

print(f"1. Loaded {len(biodiv_images)} biodiversity datasets")
for i, img in enumerate(biodiv_images):
    proj = img.projection().getInfo()
    print(f"   {biodiv_configs[i]['name']}: {proj['transform'][0]}°")

# Step 2: Process with recursive resampler
print("\n2. Processing with RecursiveResampler...")
resampler = RecursiveResampler(
    base_resolution=base_resolution,
    outputs=resolution_outputs,
    target_crs='EPSG:4326'
)

biodiv_results = resampler.process_collection(biodiv_images, biodiv_configs)
biodiv_8x = biodiv_results.get('8x', [])

print(f"   Got {len(biodiv_8x)} images at 8x")
for i, img in enumerate(biodiv_8x):
    proj = img.projection().getInfo()
    print(f"   After resampling {i}: {proj['transform'][0]}°")

# Step 3: Load climate
print("\n3. Loading climate data...")
merger_config = {
    'masking': {
        'enabled': config.masking.enabled,
        'latitude_bounds': config.masking.latitude_bounds
    }
}
merger = GEEMerger(config=merger_config)
climate = merger.load_worldclim_v2(['bio01', 'bio12'])
climate_proj = climate.projection().getInfo()
print(f"   Climate original: {climate_proj['transform'][0]}°")

# Step 4: Merge with climate (as orchestrator does)
print("\n4. Merging with climate...")
factor = 8
scale_meters = base_resolution * factor * 111320

merged = merger.merge_datasets(
    biodiv_8x,
    climate,
    'addBands',
    mask_ocean=False,
    include_elevation=False,
    scale=scale_meters
)

# Check merged projection
print("\n5. Checking merged projection...")
try:
    merged_proj = merged.projection().getInfo()
    print(f"   ERROR: Got single projection {merged_proj['transform'][0]}°")
except Exception as e:
    # This likely means bands have different projections
    print(f"   Expected error (bands have different projections): {str(e)[:100]}")
    
    # Check each band
    print("\n   Checking individual bands:")
    band_names = merged.bandNames().getInfo()
    for band in band_names[:5]:  # Check first 5 bands
        band_proj = merged.select(band).projection().getInfo()
        print(f"   {band}: {band_proj['transform'][0]}°")

# Step 6: Try export
print("\n6. Testing export...")
export_region = ee.Geometry.BBox(-10, -10, 10, 10)  # Small test region

# What the export manager would do
export_task = ee.batch.Export.image.toCloudStorage(
    image=merged,
    description='debug_full_chain',
    bucket='biodiv-gee-yl998',
    fileNamePrefix='test_8x/debug_chain',
    region=export_region,
    scale=scale_meters,  # 8x scale
    crs='EPSG:4326',
    fileFormat='GeoTIFF',
    maxPixels=1e13
)

print(f"   Starting export with scale={scale_meters:.0f}m...")
export_task.start()

# Wait and check
import time
time.sleep(5)
task_info = ee.data.getTaskStatus(export_task.id)[0]
state = task_info.get('state', 'UNKNOWN')
error = task_info.get('error_message', '')

print(f"   Task status: {state}")
if error:
    print(f"   ERROR: {error}")
    if "Too many input pixels" in error:
        print("\n❌ CONFIRMED: Projection is lost during merge!")
else:
    print("   ✅ Export started successfully")