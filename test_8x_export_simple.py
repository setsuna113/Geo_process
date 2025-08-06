#!/usr/bin/env python
"""Simple test of 8x export with proper downsampling"""

import ee
from src.gee_integration.config import GEEConfig
from src.gee_integration.auth import GEEAuthenticator

# Initialize
config = GEEConfig.from_yaml('config/gee_config.yml')
auth = GEEAuthenticator(config.__dict__)
auth.authenticate(
    method=config.auth_method,
    service_account_key=config.service_account_key,
    project_id=config.project_id
)

print("Testing 8x export with proper recursive downsampling...")

# Load a test asset
test_asset = 'projects/earthengine-legacy/assets/users/geebiodiversity/production/animals'
image = ee.Image(test_asset).rename('test')

print(f"Loaded test image: {test_asset}")

# Get base resolution
base_resolution = 0.016666666666667  # degrees
base_scale = base_resolution * 111320  # meters

print(f"Base resolution: {base_resolution}° (~{base_scale:.0f}m)")

# Step 1: Align to base resolution
base_projection = ee.Projection('EPSG:4326').atScale(base_scale)
aligned = image.reproject(base_projection)

print("Step 1: Aligned to base resolution")

# Step 2: Apply 3 recursive 2x downsamples to get 8x
current = aligned
for i in range(3):  # 3 steps: 1x -> 2x -> 4x -> 8x
    # Get current scale
    current_scale = base_scale * (2 ** i)
    target_scale = current_scale * 2
    
    print(f"Step {i+2}: Downsampling from {current_scale:.0f}m to {target_scale:.0f}m")
    
    # Apply 2x downsample
    downsampled = current.reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=4  # 2x2 pixels
    )
    
    # Reproject to target scale
    target_projection = ee.Projection('EPSG:4326').atScale(target_scale)
    current = downsampled.reproject(target_projection)

# Now we have 8x downsampled image
final_scale = base_scale * 8
print(f"\nFinal image scale: {final_scale:.0f}m ({base_resolution * 8:.6f}°)")

# Check projection
final_proj = current.projection().getInfo()
print(f"Final projection transform: {final_proj['transform'][:2]}")

# Test export parameters
test_region = ee.Geometry.BBox(-180, -60, 180, 75)
print(f"\nTest export region: Global (-180 to 180, -60 to 75)")

# Export task
export_task = ee.batch.Export.image.toCloudStorage(
    image=current,
    description='test_8x_export',
    bucket='biodiv-gee-yl998',
    fileNamePrefix='test_8x/test',
    region=test_region,
    scale=final_scale,  # Use the 8x scale
    crs='EPSG:4326',
    fileFormat='GeoTIFF',
    maxPixels=1e13
)

print(f"\nStarting export with scale={final_scale:.0f}m...")
export_task.start()
print(f"Export task started: {export_task.id}")
print("\nCheck status with:")
print(f"  earthengine task info {export_task.id}")