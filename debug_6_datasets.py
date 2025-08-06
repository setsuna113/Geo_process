#!/usr/bin/env python
"""Debug why 6 datasets fail but 2 work"""

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

print("Testing with ALL 6 biodiversity datasets...")
print("=" * 60)

# Setup
resolution_outputs = {'1x': False, '2x': False, '4x': False, '8x': True}
base_resolution = 0.016666666666667

# Load ALL 6 datasets (exactly as orchestrator does)
biodiv_images = []
biodiv_configs = []

# These are the actual assets from the config
datasets = [
    ('animals', 'animal_richness'),
    ('plants', 'plant_richness'),
    ('am_fungi', 'am_fungi_richness'),
    ('am_fungi_cv', 'am_fungi_cv'),
    ('ecm_fungi', 'ecm_fungi_richness'),
    ('ecm_fungi_cv', 'ecm_fungi_cv')
]

print("1. Loading datasets...")
for name, band_name in datasets:
    try:
        asset_id = f'projects/earthengine-legacy/assets/users/geebiodiversity/production/{name}'
        image = ee.Image(asset_id)
        
        # Check if asset exists
        image.getInfo()  # This will fail if asset doesn't exist
        
        image = image.rename([band_name])
        biodiv_images.append(image)
        biodiv_configs.append({
            'name': name,
            'band_name': band_name,
            'data_type': 'continuous',
            'resampling': 'max' if 'cv' not in name else 'rms'
        })
        
        proj = image.projection().getInfo()
        print(f"   ✓ {name}: {proj['transform'][0]}°")
    except Exception as e:
        print(f"   ✗ {name}: Failed to load - {str(e)[:50]}")

print(f"\nLoaded {len(biodiv_images)} datasets successfully")

if len(biodiv_images) == 0:
    print("No datasets loaded! Exiting.")
    exit(1)

# Process with recursive resampler
print(f"\n2. Processing {len(biodiv_images)} datasets with RecursiveResampler...")
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
    print(f"   Dataset {i}: {proj['transform'][0]}° (~{proj['transform'][0] * 111320:.0f}m)")

# Merge biodiversity datasets
print(f"\n3. Merging {len(biodiv_8x)} biodiversity datasets...")
if len(biodiv_8x) > 1:
    merged_biodiv = biodiv_8x[0]
    for img in biodiv_8x[1:]:
        merged_biodiv = merged_biodiv.addBands(img)
else:
    merged_biodiv = biodiv_8x[0]

# Check merged biodiversity projection
try:
    merged_biodiv_proj = merged_biodiv.projection().getInfo()
    print(f"   Single projection: {merged_biodiv_proj['transform'][0]}°")
except Exception as e:
    print(f"   Multiple projections (expected): {str(e)[:50]}")
    # Check first band
    first_band_proj = merged_biodiv.select(0).projection().getInfo()
    print(f"   First band: {first_band_proj['transform'][0]}°")

# Now add climate (as orchestrator does)
print("\n4. Adding climate data...")
merger = GEEMerger(config={'masking': {'enabled': False}})
climate = merger.load_worldclim_v2(['bio01', 'bio12'])

# This is what the orchestrator does
factor = 8
scale_meters = base_resolution * factor * 111320

print(f"   Merging with GEEMerger.merge_datasets...")
print(f"   Scale parameter: {scale_meters:.0f}m")

try:
    merged_full = merger.merge_datasets(
        biodiv_8x,
        climate,
        'addBands',
        mask_ocean=False,
        include_elevation=False,
        scale=scale_meters
    )
    print("   ✓ Merge successful")
    
    # Check projection of merged
    try:
        final_proj = merged_full.select(0).projection().getInfo()
        print(f"   Final projection: {final_proj['transform'][0]}°")
    except Exception as e:
        print(f"   Error checking projection: {str(e)[:100]}")
        
except Exception as e:
    print(f"   ✗ Merge failed: {str(e)[:200]}")

# Test export with small region first
print("\n5. Testing export with small region...")
test_region = ee.Geometry.BBox(-10, -10, 10, 10)

export_task = ee.batch.Export.image.toCloudStorage(
    image=merged_full if 'merged_full' in locals() else merged_biodiv,
    description='debug_6_datasets_small',
    bucket='biodiv-gee-yl998',
    fileNamePrefix='test_8x/debug_6_datasets_small',
    region=test_region,
    scale=scale_meters,
    crs='EPSG:4326',
    fileFormat='GeoTIFF',
    maxPixels=1e13
)

export_task.start()
print(f"   Task started: {export_task.id}")

# Check status
import time
time.sleep(5)
task_info = ee.data.getTaskStatus(export_task.id)[0]
state = task_info.get('state', 'UNKNOWN')
error = task_info.get('error_message', '')

if error:
    print(f"   ❌ FAILED: {error}")
else:
    print(f"   ✅ Status: {state}")

# Now test with global region (as orchestrator does)
print("\n6. Testing export with global region (-180 to 180, -60 to 75)...")
global_region = ee.Geometry.BBox(-180, -60, 180, 75)

export_task_global = ee.batch.Export.image.toCloudStorage(
    image=merged_full if 'merged_full' in locals() else merged_biodiv,
    description='debug_6_datasets_global',
    bucket='biodiv-gee-yl998',
    fileNamePrefix='test_8x/debug_6_datasets_global',
    region=global_region,
    scale=scale_meters,
    crs='EPSG:4326',
    fileFormat='GeoTIFF',
    maxPixels=1e13
)

export_task_global.start()
print(f"   Task started: {export_task_global.id}")

# Check status
time.sleep(5)
task_info = ee.data.getTaskStatus(export_task_global.id)[0]
state = task_info.get('state', 'UNKNOWN')
error = task_info.get('error_message', '')

if error:
    print(f"   ❌ FAILED: {error}")
    if "Too many input pixels" in error:
        print("\n   CONFIRMED: The issue is with 6 datasets at global scale!")
else:
    print(f"   ✅ Status: {state}")