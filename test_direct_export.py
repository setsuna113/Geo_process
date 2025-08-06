#!/usr/bin/env python
"""Test direct export with orchestrator's exact steps"""

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

print("Testing exact orchestrator steps...")

# Simulate export step
resolution_outputs = {'1x': False, '2x': False, '4x': False, '8x': True}
base_resolution = 0.016666666666667

# Load images exactly as orchestrator does
upload_results = {
    'animals': {'success': True, 'asset_id': 'projects/earthengine-legacy/assets/users/geebiodiversity/production/animals'},
}

biodiv_images = []
biodiv_configs = []

dataset_config = {
    'name': 'animals',
    'band_name': 'animal_richness',
    'data_type': 'continuous',
    'resampling': 'max'
}

asset_id = upload_results['animals']['asset_id']
image = ee.Image(asset_id)
image = image.rename([dataset_config['band_name']])
biodiv_images.append(image)
biodiv_configs.append(dataset_config)

print(f"Loaded {dataset_config['name']}")

# Check original projection
orig_proj = image.projection().getInfo()
print(f"Original projection scale: {orig_proj['transform'][0]}° (~{orig_proj['transform'][0] * 111320:.0f}m)")

# Process exactly as orchestrator does
res_name = '8x'
print(f"\nProcessing {res_name} for export...")

# Initialize resampler (EXACTLY as orchestrator does)
resampler = RecursiveResampler(
    base_resolution=base_resolution,
    outputs=resolution_outputs,
    target_crs='EPSG:4326'
)

# Process biodiversity
biodiv_results = resampler.process_collection(biodiv_images, biodiv_configs)
biodiv_resampled = biodiv_results.get(res_name, [])

print(f"Got {len(biodiv_resampled)} resampled images")

if biodiv_resampled:
    merged = biodiv_resampled[0]  # Only one image
    
    # Check merged projection
    merged_proj = merged.projection().getInfo()
    print(f"Merged projection scale: {merged_proj['transform'][0]}° (~{merged_proj['transform'][0] * 111320:.0f}m)")
    
    # Calculate export parameters (EXACTLY as orchestrator/export_manager does)
    factor = 8
    resolution_config = {
        'resolution': base_resolution * factor,
        'description': f'{factor}x downsampled'
    }
    scale_meters = resolution_config.get('resolution', 0.016667) * 111320
    
    print(f"\nExport parameters:")
    print(f"  Factor: {factor}")
    print(f"  Scale for export: {scale_meters:.0f}m")
    
    # Test export region
    export_region = ee.Geometry.BBox(-180, -60, 180, 75)
    
    print(f"\nAttempting export...")
    print(f"  Region: Global (-180 to 180, -60 to 75)")
    print(f"  Scale: {scale_meters:.0f}m")
    
    # THIS IS THE EXACT EXPORT CALL FROM EXPORT MANAGER
    export_task = ee.batch.Export.image.toCloudStorage(
        image=merged,
        description='test_orchestrator_8x',
        bucket='biodiv-gee-yl998',
        fileNamePrefix='test_8x/orchestrator_test',
        region=export_region,
        scale=scale_meters,  # This is what export manager uses
        crs='EPSG:4326',
        fileFormat='GeoTIFF',
        maxPixels=1e13
    )
    
    print(f"\nStarting export task...")
    export_task.start()
    print(f"Task ID: {export_task.id}")
    
    # Wait and check status
    import time
    time.sleep(5)
    
    # Check status
    task_info = ee.data.getTaskStatus(export_task.id)[0]
    state = task_info.get('state', 'UNKNOWN')
    error = task_info.get('error_message', '')
    
    print(f"\nTask status: {state}")
    if error:
        print(f"ERROR: {error}")
        print("\n⚠️ Export failed! The orchestrator approach doesn't work.")
    else:
        print("✅ Export started successfully!")
else:
    print("No resampled images returned!")