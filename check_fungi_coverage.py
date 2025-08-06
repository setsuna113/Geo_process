#!/usr/bin/env python
"""Check the actual data coverage of SPUN fungi SDM outputs."""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

def check_coverage(file_path, name):
    """Check data coverage for a raster file."""
    with rasterio.open(file_path) as src:
        # Read the data
        data = src.read(1)
        nodata = src.nodata
        
        # Calculate coverage
        total_pixels = data.size
        valid_pixels = np.sum((data != nodata) & np.isfinite(data))
        coverage = valid_pixels / total_pixels * 100
        
        # Get value range
        valid_data = data[(data != nodata) & np.isfinite(data)]
        
        print(f"\n{name}:")
        print(f"  Shape: {data.shape}")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Valid pixels: {valid_pixels:,}")
        print(f"  Coverage: {coverage:.2f}%")
        
        if len(valid_data) > 0:
            print(f"  Value range: {valid_data.min():.2f} - {valid_data.max():.2f}")
            print(f"  Mean (valid only): {valid_data.mean():.2f}")
            print(f"  Median (valid only): {np.median(valid_data):.2f}")
        
        # Create a quick coverage map
        coverage_map = np.where((data != nodata) & np.isfinite(data), 1, 0)
        
        return coverage_map, coverage, valid_data

# Check both fungi types
am_path = "/maps/spun/AM_fungi/AM_Fungi_Richness_Predicted.tif"
ecm_path = "/maps/spun/EcM_fungi/EcM_Fungi_Richness_Predicted.tif"

print("=" * 60)
print("SPUN FUNGI SDM COVERAGE ANALYSIS")
print("=" * 60)

# Check AM fungi
am_coverage_map, am_coverage, am_data = check_coverage(am_path, "AM Fungi")

# Check EcM fungi  
ecm_coverage_map, ecm_coverage, ecm_data = check_coverage(ecm_path, "EcM Fungi")

# Check overlap
with rasterio.open(am_path) as am_src, rasterio.open(ecm_path) as ecm_src:
    am_full = am_src.read(1)
    ecm_full = ecm_src.read(1)
    
    am_valid = (am_full != am_src.nodata) & np.isfinite(am_full)
    ecm_valid = (ecm_full != ecm_src.nodata) & np.isfinite(ecm_full)
    
    both_valid = am_valid & ecm_valid
    either_valid = am_valid | ecm_valid
    
    print(f"\nCombined Coverage:")
    print(f"  Both AM and EcM: {np.sum(both_valid)/am_full.size*100:.2f}%")
    print(f"  Either AM or EcM: {np.sum(either_valid)/am_full.size*100:.2f}%")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# AM coverage
axes[0, 0].imshow(am_coverage_map[::100, ::100], cmap='Greens', interpolation='nearest')
axes[0, 0].set_title(f'AM Fungi Coverage ({am_coverage:.1f}%)')
axes[0, 0].axis('off')

# EcM coverage
axes[0, 1].imshow(ecm_coverage_map[::100, ::100], cmap='Blues', interpolation='nearest')
axes[0, 1].set_title(f'EcM Fungi Coverage ({ecm_coverage:.1f}%)')
axes[0, 1].axis('off')

# Combined coverage
combined = np.zeros((am_coverage_map.shape[0], am_coverage_map.shape[1], 3))
combined[:,:,1] = am_coverage_map  # Green for AM
combined[:,:,2] = ecm_coverage_map  # Blue for EcM
combined[:,:,0] = am_coverage_map * ecm_coverage_map  # Red for overlap

axes[1, 0].imshow(combined[::100, ::100], interpolation='nearest')
axes[1, 0].set_title('Combined (Green=AM, Blue=EcM, Purple=Both)')
axes[1, 0].axis('off')

# Coverage by latitude
lat_bins = np.linspace(-90, 90, 37)  # 5-degree bins
lat_coverage = []
for i in range(len(lat_bins)-1):
    lat_start = int((90 - lat_bins[i+1]) / 180 * am_coverage_map.shape[0])
    lat_end = int((90 - lat_bins[i]) / 180 * am_coverage_map.shape[0])
    
    am_lat_coverage = np.mean(am_coverage_map[lat_start:lat_end])
    ecm_lat_coverage = np.mean(ecm_coverage_map[lat_start:lat_end])
    
    lat_coverage.append((lat_bins[i], am_lat_coverage, ecm_lat_coverage))

lat_coverage = np.array(lat_coverage)
axes[1, 1].plot(lat_coverage[:, 0], lat_coverage[:, 1] * 100, 'g-', label='AM', linewidth=2)
axes[1, 1].plot(lat_coverage[:, 0], lat_coverage[:, 2] * 100, 'b-', label='EcM', linewidth=2)
axes[1, 1].set_xlabel('Latitude')
axes[1, 1].set_ylabel('Coverage %')
axes[1, 1].set_title('Coverage by Latitude')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/yl998/dev/geo/fungi_coverage_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to fungi_coverage_analysis.png")

# Also check the coefficient of variation files
print("\n" + "=" * 60)
print("COEFFICIENT OF VARIATION (UNCERTAINTY) ANALYSIS")
print("=" * 60)

am_cv_path = "/maps/spun/AM_fungi/AM_Fungi_Richness_CoeffVar.tif"
ecm_cv_path = "/maps/spun/EcM_fungi/EcM_Fungi_Richness_CoeffVar.tif"

with rasterio.open(am_cv_path) as src:
    am_cv = src.read(1)
    am_cv_valid = am_cv[(am_cv != src.nodata) & np.isfinite(am_cv) & am_valid]
    if len(am_cv_valid) > 0:
        print(f"\nAM Fungi CV (where predictions exist):")
        print(f"  Mean CV: {am_cv_valid.mean():.3f}")
        print(f"  Median CV: {np.median(am_cv_valid):.3f}")
        print(f"  90th percentile CV: {np.percentile(am_cv_valid, 90):.3f}")

with rasterio.open(ecm_cv_path) as src:
    ecm_cv = src.read(1)
    ecm_cv_valid = ecm_cv[(ecm_cv != src.nodata) & np.isfinite(ecm_cv) & ecm_valid]
    if len(ecm_cv_valid) > 0:
        print(f"\nEcM Fungi CV (where predictions exist):")
        print(f"  Mean CV: {ecm_cv_valid.mean():.3f}")
        print(f"  Median CV: {np.median(ecm_cv_valid):.3f}")
        print(f"  90th percentile CV: {np.percentile(ecm_cv_valid, 90):.3f}")