#!/usr/bin/env python
"""Test global export to see why it fails"""

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

print("Testing GLOBAL export at 8x resolution...")

# Quick setup - just one dataset
image = ee.Image('projects/earthengine-legacy/assets/users/geebiodiversity/production/animals')
image = image.rename(['test'])

# Process to 8x
resampler = RecursiveResampler(
    base_resolution=0.016666666666667,
    outputs={'1x': False, '2x': False, '4x': False, '8x': True},
    target_crs='EPSG:4326'
)

results = resampler.process_dataset(image, {
    'name': 'test',
    'band_name': 'test',
    'resampling': 'max'
})

image_8x = results['8x']
proj = image_8x.projection().getInfo()
print(f"Image at 8x: {proj['transform'][0]}° (~{proj['transform'][0] * 111320:.0f}m)")

# Test different export regions
regions = [
    ("Small (20° x 20°)", ee.Geometry.BBox(-10, -10, 10, 10)),
    ("Medium (90° x 45°)", ee.Geometry.BBox(-45, -22.5, 45, 22.5)),
    ("Large (180° x 90°)", ee.Geometry.BBox(-90, -45, 90, 45)),
    ("Global masked (-180 to 180, -60 to 75)", ee.Geometry.BBox(-180, -60, 180, 75)),
    ("Full Global (-180 to 180, -90 to 90)", ee.Geometry.BBox(-180, -90, 180, 90))
]

scale_meters = 0.133333 * 111320  # 8x scale

for name, region in regions:
    print(f"\nTesting {name}...")
    
    # Calculate approximate pixel count
    bounds = region.bounds().getInfo()['coordinates'][0]
    width_deg = bounds[2][0] - bounds[0][0]
    height_deg = bounds[2][1] - bounds[0][1]
    pixels_x = int(width_deg / 0.133333)
    pixels_y = int(height_deg / 0.133333)
    total_pixels = pixels_x * pixels_y
    
    print(f"  Region: {width_deg}° x {height_deg}°")
    print(f"  Estimated pixels: {pixels_x} x {pixels_y} = {total_pixels:,}")
    
    # Try export
    export_task = ee.batch.Export.image.toCloudStorage(
        image=image_8x,
        description=f'test_8x_{name.split()[0].lower()}',
        bucket='biodiv-gee-yl998',
        fileNamePrefix=f'test_8x/region_{name.split()[0].lower()}',
        region=region,
        scale=scale_meters,
        crs='EPSG:4326',
        fileFormat='GeoTIFF',
        maxPixels=1e13
    )
    
    export_task.start()
    print(f"  Task started: {export_task.id}")
    
    # Quick check
    import time
    time.sleep(3)
    task_info = ee.data.getTaskStatus(export_task.id)[0]
    state = task_info.get('state', 'UNKNOWN')
    error = task_info.get('error_message', '')
    
    if error:
        print(f"  ❌ FAILED: {error}")
    else:
        print(f"  ✅ Status: {state}")