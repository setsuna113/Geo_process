#!/usr/bin/env python
"""Test if merging resets projection"""

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

print("Testing if merging resets projection...")

# Setup
resolution_outputs = {'1x': False, '2x': False, '4x': False, '8x': True}
base_resolution = 0.016666666666667

# Load TWO datasets
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

print(f"Loaded {len(biodiv_images)} datasets")

# Process with recursive resampler
resampler = RecursiveResampler(
    base_resolution=base_resolution,
    outputs=resolution_outputs,
    target_crs='EPSG:4326'
)

biodiv_results = resampler.process_collection(biodiv_images, biodiv_configs)
biodiv_resampled = biodiv_results.get('8x', [])

print(f"Got {len(biodiv_resampled)} resampled images")

# Check projections BEFORE merging
for i, img in enumerate(biodiv_resampled):
    proj = img.projection().getInfo()
    scale = proj['transform'][0] if proj['transform'] else None
    print(f"Image {i} before merge: {scale}° (~{scale * 111320:.0f}m)")

# Now merge them (as orchestrator does)
if len(biodiv_resampled) > 1:
    print("\nMerging images...")
    merged = biodiv_resampled[0]
    for img in biodiv_resampled[1:]:
        merged = merged.addBands(img)
else:
    merged = biodiv_resampled[0]

# Check projection AFTER merging
merged_proj = merged.projection().getInfo()
merged_scale = merged_proj['transform'][0] if merged_proj['transform'] else None
print(f"\nMerged projection: {merged_scale}° (~{merged_scale * 111320:.0f}m)")

# Now test with GEEMerger (as orchestrator does with climate)
print("\n--- Testing with GEEMerger ---")
merger_config = {
    'masking': {
        'enabled': config.masking.enabled,
        'latitude_bounds': config.masking.latitude_bounds
    }
}
merger = GEEMerger(config=merger_config)

# Simulate merging with climate
factor = 8
scale_meters = base_resolution * factor * 111320

# Try merger.merge_datasets (this is what orchestrator uses)
# First create a dummy climate image
climate = ee.Image.constant(20).rename('temperature')
# Reproject climate to match
climate_proj = ee.Projection('EPSG:4326').atScale(scale_meters)
climate = climate.reproject(climate_proj)

print(f"Climate projection: {climate.projection().getInfo()['transform'][0]}°")

# Now merge using GEEMerger
merged_with_climate = merger.merge_datasets(
    biodiv_resampled,
    climate,
    'addBands',
    mask_ocean=False,
    include_elevation=False,
    scale=scale_meters
)

# Check final projection
final_proj = merged_with_climate.projection().getInfo()
final_scale = final_proj['transform'][0] if final_proj['transform'] else None
print(f"\nFinal merged projection: {final_scale}° (~{final_scale * 111320:.0f}m)")

if abs(final_scale - (base_resolution * 8)) > 0.001:
    print("\n⚠️ PROJECTION LOST DURING MERGE!")
    print("This is why the export fails!")