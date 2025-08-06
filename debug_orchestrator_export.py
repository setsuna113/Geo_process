#!/usr/bin/env python
"""Debug why orchestrator export is failing"""

import ee
from src.gee_integration.config import GEEConfig
from src.gee_integration.auth import GEEAuthenticator
from src.gee_integration.processing.recursive_resampler import RecursiveResampler
from src.gee_integration.processing import GEEMerger

# Initialize
config = GEEConfig.from_yaml('config/gee_config.yml')

# Authenticate
auth = GEEAuthenticator(config.__dict__)
auth.authenticate(
    method=config.auth_method,
    service_account_key=config.service_account_key,
    project_id=config.project_id
)

print("Simulating orchestrator export step...")

# Get resolution settings
resolution_outputs = {'1x': False, '2x': False, '4x': False, '8x': True}
base_resolution = config.resolution_settings.get('base_resolution', 0.016666666666667)

print(f"Base resolution: {base_resolution}°")
print(f"Outputs: {resolution_outputs}")

# Load biodiversity images (just first 2)
upload_results = {
    'animals': {'success': True, 'asset_id': 'projects/earthengine-legacy/assets/users/geebiodiversity/production/animals'},
    'plants': {'success': True, 'asset_id': 'projects/earthengine-legacy/assets/users/geebiodiversity/production/plants'}
}

biodiv_images = []
biodiv_configs = []

for dataset in config.biodiversity_datasets[:2]:
    if dataset.name in upload_results:
        asset_id = upload_results[dataset.name]['asset_id']
        image = ee.Image(asset_id)
        image = image.rename([dataset.band_name])
        biodiv_images.append(image)
        biodiv_configs.append(dataset.__dict__)

# This is what the orchestrator does in export step
for res_name, should_export in resolution_outputs.items():
    if not should_export:
        continue
    
    print(f"\nProcessing {res_name} for export...")
    
    # Initialize resampler (this is what orchestrator does)
    resampler = RecursiveResampler(
        base_resolution=base_resolution,
        outputs=resolution_outputs,
        target_crs='EPSG:4326'
    )
    
    # Process biodiversity
    biodiv_results = resampler.process_collection(biodiv_images, biodiv_configs)
    biodiv_resampled = biodiv_results.get(res_name, [])
    
    print(f"  Got {len(biodiv_resampled)} resampled images")
    
    if biodiv_resampled:
        # Check first image projection
        img = biodiv_resampled[0]
        proj = img.projection().getInfo()
        scale = proj['transform'][0] if proj['transform'] else None
        print(f"  First image scale: {scale}° (~{scale * 111320:.0f}m)" if scale else "  Unknown scale")
        
        # Merge
        if len(biodiv_resampled) > 1:
            merged = biodiv_resampled[0]
            for img in biodiv_resampled[1:]:
                merged = merged.addBands(img)
        else:
            merged = biodiv_resampled[0]
        
        # Check merged projection
        merged_proj = merged.projection().getInfo()
        merged_scale = merged_proj['transform'][0] if merged_proj['transform'] else None
        print(f"  Merged scale: {merged_scale}° (~{merged_scale * 111320:.0f}m)" if merged_scale else "  Unknown scale")
        
        # This is what export manager calculates
        factor = int(res_name[:-1]) if res_name.endswith('x') else 1
        resolution_config = {
            'resolution': base_resolution * factor,
            'description': f'{factor}x downsampled'
        }
        scale_meters = resolution_config.get('resolution', 0.016667) * 111320
        
        print(f"\n  Export parameters:")
        print(f"    Factor: {factor}")
        print(f"    Resolution config: {resolution_config['resolution']}°")
        print(f"    Export scale: {scale_meters:.0f}m")
        
        # Check if scales match
        if merged_scale:
            expected_scale = base_resolution * factor
            actual_scale = merged_scale
            print(f"\n  Scale comparison:")
            print(f"    Expected: {expected_scale:.6f}°")
            print(f"    Actual: {actual_scale:.6f}°")
            print(f"    Match: {'YES' if abs(expected_scale - actual_scale) < 0.001 else 'NO'}")
            
            if abs(expected_scale - actual_scale) > 0.001:
                print(f"    ⚠️ MISMATCH! Image not properly downsampled!")
    else:
        print(f"  No images returned for {res_name}!")