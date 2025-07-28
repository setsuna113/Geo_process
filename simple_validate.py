#!/usr/bin/env python3
"""Simple validation by comparing NetCDF with CSV directly."""

import sys
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from random import randint

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def validate_data_match():
    """Compare random samples from NetCDF with CSV."""
    
    # Find latest files
    nc_files = sorted(Path("outputs").glob("*/merged_dataset.nc"), 
                     key=lambda p: p.stat().st_mtime, reverse=True)
    csv_files = sorted(Path("outputs").glob("*/merged_data*_valid_only.csv"), 
                      key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not nc_files or not csv_files:
        print("No files found")
        return
    
    nc_path = nc_files[0]
    csv_path = csv_files[0]
    
    print(f"NetCDF: {nc_path} ({nc_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"CSV: {csv_path} ({csv_path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Load NetCDF
    print("\nLoading NetCDF...")
    ds = xr.open_dataset(nc_path)
    print(f"Dimensions: {dict(ds.sizes)}")
    print(f"Variables: {list(ds.data_vars)}")
    
    # Load CSV header to check structure
    print("\nChecking CSV structure...")
    csv_sample = pd.read_csv(csv_path, nrows=5)
    print(f"CSV columns: {list(csv_sample.columns)}")
    print(f"CSV sample:\n{csv_sample}")
    
    # Random validation samples
    n_samples = 50
    print(f"\n=== Validating {n_samples} random points ===")
    
    matches = 0
    mismatches = []
    
    for i in range(n_samples):
        # Random indices
        y_idx = randint(0, ds.sizes['y'] - 1)
        x_idx = randint(0, ds.sizes['x'] - 1)
        
        # Get values from NetCDF
        nc_values = {}
        has_data = False
        for var in ds.data_vars:
            value = ds[var].isel(y=y_idx, x=x_idx).values.item()
            nc_values[var] = value
            if not np.isnan(value):
                has_data = True
        
        if not has_data:
            continue  # Skip if all NaN
        
        # Find in CSV
        csv_row = csv_sample[(csv_sample['y'] == y_idx) & (csv_sample['x'] == x_idx)]
        
        if csv_row.empty:
            # Try loading a chunk around this point
            chunk_start = max(0, i * 1000000)
            chunk = pd.read_csv(csv_path, skiprows=chunk_start, nrows=100000)
            csv_row = chunk[(chunk['y'] == y_idx) & (chunk['x'] == x_idx)]
        
        if not csv_row.empty:
            # Compare values
            match = True
            for var in ds.data_vars:
                nc_val = nc_values[var]
                csv_val = csv_row.iloc[0][var]
                
                if np.isnan(nc_val) and np.isnan(csv_val):
                    continue
                elif abs(nc_val - csv_val) > 0.0001:
                    match = False
                    mismatches.append({
                        'location': f"({y_idx}, {x_idx})",
                        'variable': var,
                        'nc_value': nc_val,
                        'csv_value': csv_val
                    })
            
            if match:
                matches += 1
        
        if (i + 1) % 10 == 0:
            print(f"Checked {i + 1} samples: {matches} matches found")
    
    print(f"\n=== Results ===")
    print(f"Total matches: {matches}/{n_samples}")
    
    if mismatches:
        print(f"\nFirst 5 mismatches:")
        for m in mismatches[:5]:
            print(f"  {m['location']} {m['variable']}: NC={m['nc_value']:.4f}, CSV={m['csv_value']:.4f}")
    
    # Check if CSV uses indices or coordinates
    print("\n=== Coordinate System Check ===")
    print(f"NetCDF y range: [0, {ds.sizes['y']-1}]")
    print(f"NetCDF x range: [0, {ds.sizes['x']-1}]")
    print(f"CSV y range: [{csv_sample['y'].min()}, {csv_sample['y'].max()}]")
    print(f"CSV x range: [{csv_sample['x'].min()}, {csv_sample['x'].max()}]")
    
    if csv_sample['y'].max() < 200:  # Likely lat/lon
        print("\nCSV appears to use lat/lon coordinates, not indices!")
        print("This explains why direct index matching fails.")
        
        # Try coordinate-based matching
        if 'lat' in ds.coords and 'lon' in ds.coords:
            print("\nTrying coordinate-based matching...")
            lat_sample = ds.coords['lat'].isel(y=y_idx).values
            lon_sample = ds.coords['lon'].isel(x=x_idx).values
            print(f"Sample NetCDF coordinate: lat={lat_sample:.4f}, lon={lon_sample:.4f}")
    
    ds.close()

if __name__ == "__main__":
    validate_data_match()