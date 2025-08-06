#!/usr/bin/env python
"""Debug script to understand 8x export issue"""

import ee
from src.gee_integration.config import GEEConfig
from src.gee_integration.auth import GEEAuthenticator
from src.gee_integration.processing.recursive_resampler import RecursiveResampler

# Initialize
config = GEEConfig.from_yaml('config/gee_config.yml')
print(f"Config loaded: {config.resolution_settings}")

# Authenticate
auth = GEEAuthenticator(config.__dict__)
auth.authenticate(
    method=config.auth_method,
    service_account_key=config.service_account_key,
    project_id=config.project_id
)
print("Authenticated with GEE")

# Create test image at base resolution
base_resolution = config.resolution_settings['base_resolution']
print(f"\nBase resolution: {base_resolution}° (~{base_resolution * 111320:.0f}m)")

# Create a simple test image
test_image = ee.Image.constant(1).rename('test')
print(f"Created test image")

# Get projection info at base resolution
base_projection = ee.Projection('EPSG:4326').atScale(base_resolution * 111320)
test_image = test_image.reproject(base_projection)

print(f"\nBase image projection:")
proj_info = test_image.projection().getInfo()
print(f"  CRS: {proj_info['crs']}")
print(f"  Transform: {proj_info['transform']}")

# Now apply recursive downsampling
resampler = RecursiveResampler(
    base_resolution=base_resolution,
    outputs={'1x': False, '2x': False, '4x': False, '8x': True},
    target_crs='EPSG:4326'
)

print(f"\nProcessing with recursive resampler...")
config_dict = {'name': 'test', 'band_name': 'test', 'resampling': 'mean'}
results = resampler.process_dataset(test_image, config_dict)

if '8x' in results:
    image_8x = results['8x']
    print(f"\n8x image created")
    
    # Check its projection
    proj_8x = image_8x.projection().getInfo()
    print(f"8x image projection:")
    print(f"  CRS: {proj_8x['crs']}")
    print(f"  Transform: {proj_8x['transform']}")
    
    # Calculate expected vs actual scale
    expected_scale = base_resolution * 8 * 111320  # 8x downsampled
    actual_scale = proj_8x['transform'][0] if proj_8x['transform'] else None
    
    print(f"\nScale comparison:")
    print(f"  Expected scale: {expected_scale:.2f}m ({base_resolution * 8:.6f}°)")
    print(f"  Actual scale: {actual_scale:.2f}m" if actual_scale else "  Actual scale: Unknown")
    
    # Now try to export with the expected scale
    print(f"\nTrying export with scale={expected_scale:.0f}m...")
    
    # Small test region
    test_region = ee.Geometry.BBox(-10, -10, 10, 10)
    
    # Check pixel count
    pixel_count = image_8x.select(0).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=test_region,
        scale=expected_scale,
        maxPixels=1e9
    )
    
    print(f"Pixel count in test region: {pixel_count.getInfo()}")
else:
    print("No 8x image produced!")