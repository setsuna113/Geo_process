#!/usr/bin/env python
"""Debug why export is failing with too many pixels"""

import ee
from src.gee_integration.config import GEEConfig  
from src.gee_integration.workflow.orchestrator import GEEWorkflowOrchestrator

# Initialize
config = GEEConfig.from_yaml('config/gee_config.yml')
orchestrator = GEEWorkflowOrchestrator('config/gee_config.yml', 'debug_8x_test')

# Authenticate
orchestrator._authenticate()

# Get the state
state = orchestrator.state
upload_results = state.get_metadata('upload_results', {})

print("Upload results available:")
for name, result in upload_results.items():
    print(f"  {name}: {result.get('asset_id', 'N/A')}")

# Now simulate the export step processing
from src.gee_integration.processing import GEEMerger
from src.gee_integration.processing.recursive_resampler import RecursiveResampler

merger_config = {
    'masking': {
        'enabled': config.masking.enabled,
        'latitude_bounds': config.masking.latitude_bounds
    }
}
merger = GEEMerger(config=merger_config)

# Get resolution settings
resolution_outputs = orchestrator._get_resolution_outputs()
base_resolution = config.resolution_settings.get('base_resolution', 0.016666666666667)

print(f"\nResolution settings:")
print(f"  Base: {base_resolution}°")
print(f"  Outputs: {resolution_outputs}")

# Load biodiversity images (just first 2 for testing)
biodiv_images = []
biodiv_configs = []

for dataset in config.biodiversity_datasets[:2]:
    if dataset.name in upload_results and upload_results[dataset.name].get('success'):
        asset_id = upload_results[dataset.name]['asset_id']
        image = ee.Image(asset_id)
        image = image.rename([dataset.band_name])
        biodiv_images.append(image)
        biodiv_configs.append(dataset.__dict__)
        
        # Check original image scale
        orig_proj = image.projection().getInfo()
        print(f"\n{dataset.name} original projection:")
        print(f"  Transform: {orig_proj['transform'][:2]}...")

# Initialize resampler
resampler = RecursiveResampler(
    base_resolution=base_resolution,
    outputs=resolution_outputs,
    target_crs='EPSG:4326'
)

# Process for 8x
print("\nProcessing with recursive resampler...")
biodiv_results = resampler.process_collection(biodiv_images, biodiv_configs)
biodiv_8x = biodiv_results.get('8x', [])

if biodiv_8x:
    print(f"Got {len(biodiv_8x)} images at 8x resolution")
    
    # Check projection of processed images
    for i, img in enumerate(biodiv_8x):
        proj = img.projection().getInfo()
        scale = proj['transform'][0] if proj['transform'] else None
        print(f"  Image {i} scale: {scale}° (~{scale * 111320:.0f}m)" if scale else f"  Image {i}: Unknown scale")
    
    # Merge
    if len(biodiv_8x) > 1:
        merged = biodiv_8x[0]
        for img in biodiv_8x[1:]:
            merged = merged.addBands(img)
    else:
        merged = biodiv_8x[0]
    
    print(f"\nMerged image bands: {merged.bandNames().getInfo()}")
    
    # Check merged projection
    merged_proj = merged.projection().getInfo()
    print(f"Merged projection transform: {merged_proj['transform'][:2]}...")
    
    # Now simulate the export
    factor = 8  # For 8x
    export_scale = base_resolution * factor * 111320
    print(f"\nExport scale calculation:")
    print(f"  Factor: {factor}")
    print(f"  Export scale: {export_scale:.0f}m ({base_resolution * factor:.6f}°)")
    
    # Test region
    test_region = ee.Geometry.BBox(-180, -60, 180, 75)  # Production bounds
    
    print("\nSimulating export...")
    print(f"  Region: Global (-180 to 180, -60 to 75)")
    print(f"  Scale: {export_scale:.0f}m")
    
    # This is what the export would do
    # Check if it would fail
    try:
        # Small test to see if export parameters work
        small_region = ee.Geometry.BBox(-10, -10, 10, 10)
        test_export = merged.select(0).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=small_region,
            scale=export_scale,
            maxPixels=1e9
        )
        count = test_export.getInfo()
        print(f"  Test export successful: {count}")
    except Exception as e:
        print(f"  Test export failed: {e}")
else:
    print("No 8x images produced!")