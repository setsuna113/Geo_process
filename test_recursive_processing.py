#!/usr/bin/env python
"""Test the full recursive processing chain"""

import ee
from src.gee_integration.config import GEEConfig
from src.gee_integration.auth import GEEAuthenticator
from src.gee_integration.processing.recursive_resampler import RecursiveResampler
from src.gee_integration.processing import GEEMerger

# Initialize
config = GEEConfig.from_yaml('config/gee_config.yml')
print(f"Testing 8x-only export configuration")

# Authenticate
auth = GEEAuthenticator(config.__dict__)
auth.authenticate(
    method=config.auth_method,
    service_account_key=config.service_account_key,
    project_id=config.project_id
)

# Load first biodiversity dataset as test
upload_results = {
    'animals': {'success': True, 'asset_id': 'projects/earthengine-legacy/assets/users/geebiodiversity/production/animals'},
    'plants': {'success': True, 'asset_id': 'projects/earthengine-legacy/assets/users/geebiodiversity/production/plants'}
}

# Load images
biodiv_images = []
biodiv_configs = []

for dataset in config.biodiversity_datasets[:2]:  # Just test with first 2
    if dataset.name in upload_results:
        asset_id = upload_results[dataset.name]['asset_id']
        image = ee.Image(asset_id)
        image = image.rename([dataset.band_name])
        biodiv_images.append(image)
        biodiv_configs.append(dataset.__dict__)
        print(f"Loaded {dataset.name}")

# Initialize recursive resampler
base_resolution = config.resolution_settings['base_resolution']
outputs = config.resolution_settings['outputs']

print(f"\nBase resolution: {base_resolution}°")
print(f"Outputs: {outputs}")

resampler = RecursiveResampler(
    base_resolution=base_resolution,
    outputs=outputs,
    target_crs='EPSG:4326'
)

# Process biodiversity through recursive downsampling
print("\nProcessing biodiversity datasets...")
biodiv_results = resampler.process_collection(biodiv_images, biodiv_configs)

# Check what we got
print(f"\nResults:")
for res_name, images in biodiv_results.items():
    print(f"  {res_name}: {len(images)} images")
    if images:
        # Check projection of first image
        proj = images[0].projection().getInfo()
        scale = proj['transform'][0] if proj['transform'] else None
        print(f"    Scale: {scale}° (~{scale * 111320:.0f}m)" if scale else "    Scale: Unknown")

# Now try merging for 8x
if '8x' in biodiv_results:
    images_8x = biodiv_results['8x']
    print(f"\nMerging {len(images_8x)} images for 8x resolution...")
    
    if len(images_8x) > 1:
        merged = images_8x[0]
        for img in images_8x[1:]:
            merged = merged.addBands(img)
    else:
        merged = images_8x[0]
    
    # Check merged projection
    merged_proj = merged.projection().getInfo()
    merged_scale = merged_proj['transform'][0] if merged_proj['transform'] else None
    print(f"Merged image scale: {merged_scale}° (~{merged_scale * 111320:.0f}m)" if merged_scale else "Merged scale: Unknown")
    
    # Calculate what scale the export should use
    expected_export_scale = base_resolution * 8 * 111320  # in meters
    print(f"\nExpected export scale: {expected_export_scale:.0f}m")
    print(f"This is {base_resolution * 8:.6f}° which matches the processed image")
    
    # Test a small export
    print("\nTesting export parameters...")
    test_region = ee.Geometry.BBox(-10, -10, 10, 10)
    
    # Count pixels at the export scale
    pixel_count = merged.select(0).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=test_region,
        scale=expected_export_scale,
        maxPixels=1e9
    ).getInfo()
    
    print(f"Pixels in 20°x20° test region at {expected_export_scale:.0f}m scale: {pixel_count}")
    
    # Check if export would work
    total_pixels = list(pixel_count.values())[0] if pixel_count else 0
    if total_pixels > 0:
        pixels_per_degree = total_pixels / 400  # 20x20 degree region
        global_pixels = pixels_per_degree * 360 * 180  # Rough estimate
        print(f"Estimated global pixels: {global_pixels:.0f}")
        print(f"Export should {'succeed' if global_pixels < 1e10 else 'fail (too many pixels)'}")
else:
    print("\nNo 8x results produced!")