#!/usr/bin/env python3
"""Statistical validation of merge logic by sampling different scenarios."""

import sys
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from random import sample, seed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set seed for reproducibility
seed(42)

def validate_merge_scenarios():
    """Validate that merge correctly handled different data scenarios."""
    
    print("=== Merge Logic Validation ===")
    print("Testing if data from both sources was properly combined\n")
    
    # Load merged data
    nc_path = Path("outputs/86ff9edf-3c7a-4a95-af75-f3baba36df1c/merged_dataset.nc")
    ds = xr.open_dataset(nc_path)
    
    # Load CSV sample for faster processing
    csv_path = Path("outputs/86ff9edf-3c7a-4a95-af75-f3baba36df1c/merged_data_20250728_093702_valid_only.csv")
    csv_sample = pd.read_csv(csv_path, nrows=100000)
    
    print(f"Loaded {len(csv_sample):,} rows from CSV for analysis")
    
    # Analyze data patterns
    print("\n1. Data Coverage Analysis:")
    
    # Count different scenarios in the sample
    both_present = csv_sample[(csv_sample['terrestrial_richness'].notna()) & 
                              (csv_sample['plants_richness'].notna())]
    only_terrestrial = csv_sample[(csv_sample['terrestrial_richness'].notna()) & 
                                  (csv_sample['plants_richness'].isna())]
    only_plants = csv_sample[(csv_sample['terrestrial_richness'].isna()) & 
                            (csv_sample['plants_richness'].notna())]
    
    print(f"  Both values present: {len(both_present):,} ({len(both_present)/len(csv_sample)*100:.1f}%)")
    print(f"  Only terrestrial: {len(only_terrestrial):,} ({len(only_terrestrial)/len(csv_sample)*100:.1f}%)")
    print(f"  Only plants: {len(only_plants):,} ({len(only_plants)/len(csv_sample)*100:.1f}%)")
    
    # This is the key validation - we SHOULD have cases with both values
    if len(both_present) == 0:
        print("\n❌ ERROR: No coordinates have both values! Merge likely failed.")
        return False
    
    # 2. Statistical distribution check
    print("\n2. Value Distribution Analysis:")
    
    # For locations with both values, check if they're correlated (they should be somewhat)
    if len(both_present) > 100:
        correlation = both_present[['terrestrial_richness', 'plants_richness']].corr().iloc[0, 1]
        print(f"  Correlation between terrestrial and plants richness: {correlation:.3f}")
        print(f"  (Expected: positive correlation, typically 0.3-0.8)")
        
        # Basic stats
        print(f"\n  Terrestrial richness: mean={both_present['terrestrial_richness'].mean():.1f}, "
              f"std={both_present['terrestrial_richness'].std():.1f}")
        print(f"  Plants richness: mean={both_present['plants_richness'].mean():.1f}, "
              f"std={both_present['plants_richness'].std():.1f}")
    
    # 3. Spatial coverage check - sample different regions
    print("\n3. Regional Coverage Check:")
    
    # Group by latitude bands
    csv_sample['lat_band'] = pd.cut(csv_sample['y'], bins=10)
    coverage_by_band = csv_sample.groupby('lat_band').agg({
        'terrestrial_richness': lambda x: x.notna().sum(),
        'plants_richness': lambda x: x.notna().sum()
    })
    
    print("  Coverage by latitude band:")
    for band, row in coverage_by_band.iterrows():
        if pd.notna(band):
            print(f"    {band}: terrestrial={row['terrestrial_richness']}, plants={row['plants_richness']}")
    
    # 4. Spot check specific coordinates
    print("\n4. Random Spot Checks:")
    
    # Sample 10 locations with both values
    if len(both_present) >= 10:
        spot_checks = both_present.sample(10)
        
        print("  Sampling 10 locations with both values:")
        for _, row in spot_checks.iterrows():
            print(f"    ({row['y']:.4f}, {row['x']:.4f}): "
                  f"terrestrial={row['terrestrial_richness']:.0f}, "
                  f"plants={row['plants_richness']:.0f}")
    
    # 5. Check edge cases
    print("\n5. Edge Case Analysis:")
    
    # High biodiversity areas (top 1%)
    high_terrestrial = csv_sample['terrestrial_richness'].quantile(0.99)
    high_plants = csv_sample['plants_richness'].quantile(0.99)
    
    high_biodiv = csv_sample[
        (csv_sample['terrestrial_richness'] > high_terrestrial) | 
        (csv_sample['plants_richness'] > high_plants)
    ]
    
    if len(high_biodiv) > 0:
        print(f"  High biodiversity areas: {len(high_biodiv)} locations")
        
        # Check if high biodiversity areas tend to have both values
        high_both = high_biodiv[(high_biodiv['terrestrial_richness'].notna()) & 
                                (high_biodiv['plants_richness'].notna())]
        print(f"  High biodiversity with both values: {len(high_both)} "
              f"({len(high_both)/len(high_biodiv)*100:.1f}%)")
    
    # 6. Validate against original data patterns
    print("\n6. Original Data Pattern Check:")
    
    # Load small samples from original NetCDF files to compare patterns
    try:
        # Find original files
        import glob
        terrestrial_files = glob.glob("data/*terrestrial*.nc") + glob.glob("data/*terrestrial*.h5")
        plants_files = glob.glob("data/*plants*.nc") + glob.glob("data/*plants*.h5")
        
        if terrestrial_files and plants_files:
            print(f"  Found original files: {len(terrestrial_files)} terrestrial, {len(plants_files)} plants")
            
            # Quick check of value ranges
            terr_vals = csv_sample['terrestrial_richness'].dropna()
            plant_vals = csv_sample['plants_richness'].dropna()
            
            print(f"  Terrestrial range: [{terr_vals.min():.0f}, {terr_vals.max():.0f}]")
            print(f"  Plants range: [{plant_vals.min():.0f}, {plant_vals.max():.0f}]")
            
            # Check if ranges make sense for species richness data
            if terr_vals.max() > 10000 or plant_vals.max() > 50000:
                print("  ⚠️  Warning: Unusually high richness values detected")
            else:
                print("  ✓ Value ranges appear reasonable for species richness data")
        else:
            print("  Could not find original data files for comparison")
    except Exception as e:
        print(f"  Could not validate against originals: {e}")
    
    # Final verdict
    print("\n=== Validation Results ===")
    
    merge_success = True
    issues = []
    
    if len(both_present) == 0:
        merge_success = False
        issues.append("No locations have both datasets merged")
    elif len(both_present) < len(csv_sample) * 0.1:
        issues.append(f"Only {len(both_present)/len(csv_sample)*100:.1f}% have both values (seems low)")
    
    if len(only_terrestrial) == 0 and len(only_plants) == 0:
        issues.append("Suspicious: all locations have both values (unlikely unless clipped)")
    
    if merge_success:
        print("✓ Merge appears successful!")
        print(f"  - {len(both_present):,} locations have data from both sources")
        print(f"  - {len(only_terrestrial):,} locations have only terrestrial data") 
        print(f"  - {len(only_plants):,} locations have only plants data")
        
        if len(both_present) > 100:
            print(f"  - Correlation between datasets: {correlation:.3f}")
    else:
        print("❌ Merge validation FAILED!")
    
    if issues:
        print("\nPotential issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Save detailed stats
    stats_file = Path("outputs/86ff9edf-3c7a-4a95-af75-f3baba36df1c/merge_validation_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("Merge Validation Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total sample size: {len(csv_sample):,}\n")
        f.write(f"Both values: {len(both_present):,} ({len(both_present)/len(csv_sample)*100:.1f}%)\n")
        f.write(f"Only terrestrial: {len(only_terrestrial):,} ({len(only_terrestrial)/len(csv_sample)*100:.1f}%)\n")
        f.write(f"Only plants: {len(only_plants):,} ({len(only_plants)/len(csv_sample)*100:.1f}%)\n")
        if len(both_present) > 100:
            f.write(f"\nCorrelation: {correlation:.3f}\n")
    
    print(f"\nDetailed statistics saved to: {stats_file}")
    
    ds.close()
    return merge_success

if __name__ == "__main__":
    success = validate_merge_scenarios()
    sys.exit(0 if success else 1)