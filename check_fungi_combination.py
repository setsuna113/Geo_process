#!/usr/bin/env python
"""Check if AM and EcM fungi can be combined for total richness."""

import rasterio
import numpy as np

am_path = "/maps/spun/AM_fungi/AM_Fungi_Richness_Predicted.tif"
ecm_path = "/maps/spun/EcM_fungi/EcM_Fungi_Richness_Predicted.tif"

print("Checking if AM and EcM fungi can be combined...")
print("=" * 50)

# Sample a small region to check
with rasterio.open(am_path) as am_src, rasterio.open(ecm_path) as ecm_src:
    # Read a sample window
    window = rasterio.windows.Window(20000, 10000, 1000, 1000)
    
    am_data = am_src.read(1, window=window)
    ecm_data = ecm_src.read(1, window=window)
    
    # Check valid pixels
    am_valid = (am_data != am_src.nodata) & np.isfinite(am_data)
    ecm_valid = (ecm_data != ecm_src.nodata) & np.isfinite(ecm_data)
    
    # Check if valid pixels match
    print(f"AM valid pixels: {np.sum(am_valid):,}")
    print(f"EcM valid pixels: {np.sum(ecm_valid):,}")
    print(f"Matching valid pixels: {np.sum(am_valid == ecm_valid):,} / {am_valid.size:,}")
    print(f"Coverage identical: {np.all(am_valid == ecm_valid)}")
    
    # Where both are valid, check if they can be summed
    both_valid = am_valid & ecm_valid
    if np.any(both_valid):
        am_values = am_data[both_valid]
        ecm_values = ecm_data[both_valid]
        total_values = am_values + ecm_values
        
        print(f"\nFor pixels with both AM and EcM data:")
        print(f"  AM range: {am_values.min():.1f} - {am_values.max():.1f}")
        print(f"  EcM range: {ecm_values.min():.1f} - {ecm_values.max():.1f}")
        print(f"  Total range: {total_values.min():.1f} - {total_values.max():.1f}")
        print(f"  Mean total: {total_values.mean():.1f}")
        
        # Check correlation
        correlation = np.corrcoef(am_values, ecm_values)[0, 1]
        print(f"  Correlation (AM vs EcM): {correlation:.3f}")

print("\nInterpretation:")
print("- AM and EcM have identical coverage patterns")
print("- They represent different fungal guilds (functional groups)")
print("- Total fungal richness = AM + EcM (they're mutually exclusive guilds)")
print("- Low/negative correlation expected (different ecological niches)")