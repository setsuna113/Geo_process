#!/usr/bin/env python3
"""Trace CSV values back to original TIF files."""

import pandas as pd
import rasterio
import numpy as np
from pathlib import Path

def trace_values_to_source():
    """Verify CSV values match original TIF files."""
    
    print("=== Tracing Values to Original Sources ===\n")
    
    # Original TIF files from config.yml
    tif_files = {
        'terrestrial_richness': '/maps/mwd24/richness/iucn-terrestrial-richness.tif',
        'plants_richness': '/maps/mwd24/richness/daru-plants-richness.tif'
    }
    
    # Load CSV sample with both values present
    csv_path = 'outputs/86ff9edf-3c7a-4a95-af75-f3baba36df1c/merged_data_20250728_093702_valid_only.csv'
    
    # Skip to middle section where we know there's diversity
    df = pd.read_csv(csv_path, skiprows=40000000, nrows=1000, header=None,
                     names=['y', 'x', 'terrestrial_richness', 'plants_richness'])
    
    # Find rows with both values
    both_present = df[(df['terrestrial_richness'].notna()) & 
                     (df['plants_richness'].notna())]
    
    if len(both_present) == 0:
        print("No rows with both values in this sample!")
        return
    
    # Sample 5 test cases
    test_cases = both_present.sample(min(5, len(both_present)), random_state=42)
    
    print(f"Testing {len(test_cases)} coordinates:\n")
    
    all_matches = True
    
    for idx, row in test_cases.iterrows():
        lat, lon = row['y'], row['x']
        csv_terr = row['terrestrial_richness']
        csv_plant = row['plants_richness']
        
        print(f"Coordinate ({lat:.4f}, {lon:.4f}):")
        print(f"  CSV values: terrestrial={csv_terr:.0f}, plants={csv_plant:.0f}")
        
        # Check each TIF file
        matches = {}
        
        for var_name, tif_path in tif_files.items():
            if not Path(tif_path).exists():
                print(f"  ❌ {var_name}: File not found at {tif_path}")
                continue
                
            try:
                with rasterio.open(tif_path) as src:
                    # Get value at coordinate
                    # rasterio uses (lon, lat) order
                    row_idx, col_idx = src.index(lon, lat)
                    
                    # Read the specific pixel
                    window = rasterio.windows.Window(col_idx, row_idx, 1, 1)
                    data = src.read(1, window=window)
                    
                    if data.size > 0:
                        tif_value = data[0, 0]
                        
                        # Handle nodata
                        if src.nodata is not None and tif_value == src.nodata:
                            tif_value = np.nan
                        
                        # Compare with CSV
                        csv_value = row[var_name]
                        
                        if np.isnan(tif_value) and np.isnan(csv_value):
                            matches[var_name] = True
                            print(f"  ✓ {var_name}: Both NaN")
                        elif not np.isnan(tif_value) and not np.isnan(csv_value):
                            if abs(tif_value - csv_value) < 0.0001:
                                matches[var_name] = True
                                print(f"  ✓ {var_name}: TIF={tif_value:.0f} matches CSV={csv_value:.0f}")
                            else:
                                matches[var_name] = False
                                print(f"  ❌ {var_name}: TIF={tif_value:.0f} ≠ CSV={csv_value:.0f}")
                        else:
                            matches[var_name] = False
                            print(f"  ❌ {var_name}: TIF={tif_value} ≠ CSV={csv_value}")
                    else:
                        print(f"  ⚠ {var_name}: No data at this coordinate")
                        
            except Exception as e:
                print(f"  ❌ {var_name}: Error reading TIF - {str(e)}")
                matches[var_name] = False
        
        # Check if both matched
        if all(matches.get(var, False) for var in tif_files.keys()):
            print(f"  ✅ ALL VALUES MATCH!")
        else:
            print(f"  ❌ Some values don't match")
            all_matches = False
        
        print()
    
    # Summary
    print("\n=== VALIDATION SUMMARY ===")
    if all_matches:
        print("✅ All tested CSV values successfully traced to original TIF files!")
        print("   The merge preserved data integrity from source to output.")
    else:
        print("❌ Some values could not be traced to original files.")
        print("   This may indicate issues with coordinate systems or resampling.")
    
    # Additional check: print TIF metadata
    print("\n=== Original TIF File Info ===")
    for var_name, tif_path in tif_files.items():
        if Path(tif_path).exists():
            try:
                with rasterio.open(tif_path) as src:
                    print(f"\n{var_name} ({Path(tif_path).name}):")
                    print(f"  CRS: {src.crs}")
                    print(f"  Bounds: {src.bounds}")
                    print(f"  Resolution: {src.res}")
                    print(f"  Shape: {src.shape}")
                    print(f"  NoData value: {src.nodata}")
            except Exception as e:
                print(f"\n{var_name}: Could not read metadata - {e}")

if __name__ == "__main__":
    trace_values_to_source()