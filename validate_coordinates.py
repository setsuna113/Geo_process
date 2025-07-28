#!/usr/bin/env python3
"""Validate data by matching coordinates between NetCDF and CSV."""

import sys
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from random import randint, sample

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def validate_with_coordinates():
    """Compare NetCDF and CSV using coordinate matching."""
    
    # Find latest files
    nc_path = sorted(Path("outputs").glob("*/merged_dataset.nc"), 
                     key=lambda p: p.stat().st_mtime, reverse=True)[0]
    csv_path = sorted(Path("outputs").glob("*/merged_data*_valid_only.csv"), 
                      key=lambda p: p.stat().st_mtime, reverse=True)[0]
    
    print(f"NetCDF: {nc_path}")
    print(f"CSV: {csv_path}")
    
    # Load NetCDF
    print("\nLoading NetCDF dataset...")
    ds = xr.open_dataset(nc_path)
    
    # Check if we have lat/lon coordinates
    if 'lat' not in ds.coords or 'lon' not in ds.coords:
        # Create coordinates from indices
        print("Creating lat/lon coordinates from indices...")
        # Assuming a global grid, adjust these values based on your actual grid
        lat_start, lat_end = 90, -90
        lon_start, lon_end = -180, 180
        
        lats = np.linspace(lat_start, lat_end, ds.sizes['y'])
        lons = np.linspace(lon_start, lon_end, ds.sizes['x'])
        
        ds = ds.assign_coords(lat=('y', lats), lon=('x', lons))
    
    print(f"NetCDF shape: y={ds.sizes['y']}, x={ds.sizes['x']}")
    print(f"Lat range: [{ds.coords['lat'].min().values:.2f}, {ds.coords['lat'].max().values:.2f}]")
    print(f"Lon range: [{ds.coords['lon'].min().values:.2f}, {ds.coords['lon'].max().values:.2f}]")
    
    # Load a sample of CSV data
    print("\nLoading CSV sample...")
    csv_sample = pd.read_csv(csv_path, nrows=10000)
    print(f"CSV sample size: {len(csv_sample)} rows")
    print(f"CSV lat range: [{csv_sample['y'].min():.2f}, {csv_sample['y'].max():.2f}]")
    print(f"CSV lon range: [{csv_sample['x'].min():.2f}, {csv_sample['x'].max():.2f}]")
    
    # Validate random points
    print("\n=== Validating random samples ===")
    n_tests = 20
    test_indices = sample(range(len(csv_sample)), min(n_tests, len(csv_sample)))
    
    matches = 0
    value_matches = 0
    errors = []
    
    for idx in test_indices:
        csv_row = csv_sample.iloc[idx]
        csv_lat = csv_row['y']
        csv_lon = csv_row['x']
        
        # Find nearest point in NetCDF
        lat_idx = np.argmin(np.abs(ds.coords['lat'].values - csv_lat))
        lon_idx = np.argmin(np.abs(ds.coords['lon'].values - csv_lon))
        
        nc_lat = ds.coords['lat'].isel(y=lat_idx).values
        nc_lon = ds.coords['lon'].isel(x=lon_idx).values
        
        # Check coordinate match
        coord_error = np.sqrt((nc_lat - csv_lat)**2 + (nc_lon - csv_lon)**2)
        
        if coord_error < 0.1:  # Within ~10km
            matches += 1
            
            # Check values
            all_match = True
            for var in ds.data_vars:
                nc_val = ds[var].isel(y=lat_idx, x=lon_idx).values.item()
                csv_val = csv_row[var]
                
                if np.isnan(nc_val) and np.isnan(csv_val):
                    continue
                elif not np.isnan(nc_val) and not np.isnan(csv_val):
                    if abs(nc_val - csv_val) > 0.0001:
                        all_match = False
                        errors.append({
                            'coord': f"({csv_lat:.2f}, {csv_lon:.2f})",
                            'var': var,
                            'nc': nc_val,
                            'csv': csv_val,
                            'diff': abs(nc_val - csv_val)
                        })
                else:
                    all_match = False
            
            if all_match:
                value_matches += 1
        
        if (idx + 1) % 5 == 0:
            print(f"  Tested {idx + 1}: {matches} coordinate matches, {value_matches} value matches")
    
    # Results
    print(f"\n=== VALIDATION RESULTS ===")
    print(f"Coordinate matches: {matches}/{n_tests} ({matches/n_tests*100:.1f}%)")
    print(f"Value matches: {value_matches}/{matches} ({value_matches/matches*100:.1f}% of coord matches)")
    
    if errors:
        print(f"\nFirst 5 value mismatches:")
        for e in errors[:5]:
            print(f"  {e['coord']} {e['var']}: NC={e['nc']:.4f}, CSV={e['csv']:.4f}, diff={e['diff']:.6f}")
    
    # Check grid alignment
    print("\n=== Grid Alignment Check ===")
    # Sample some CSV coordinates and find their NetCDF indices
    sample_coords = csv_sample[['y', 'x']].drop_duplicates().sample(min(10, len(csv_sample)))
    
    print("Sample coordinate mappings:")
    for _, row in sample_coords.iterrows():
        csv_lat, csv_lon = row['y'], row['x']
        lat_idx = np.argmin(np.abs(ds.coords['lat'].values - csv_lat))
        lon_idx = np.argmin(np.abs(ds.coords['lon'].values - csv_lon))
        nc_lat = ds.coords['lat'].isel(y=lat_idx).values
        nc_lon = ds.coords['lon'].isel(x=lon_idx).values
        
        print(f"  CSV: ({csv_lat:.4f}, {csv_lon:.4f}) -> NC indices: ({lat_idx}, {lon_idx}) "
              f"-> NC coords: ({nc_lat:.4f}, {nc_lon:.4f})")
    
    # Try to figure out the grid parameters
    print("\n=== Grid Resolution Analysis ===")
    if len(csv_sample) > 100:
        # Get unique sorted latitudes
        unique_lats = sorted(csv_sample['y'].unique())[:10]
        if len(unique_lats) > 1:
            lat_diffs = [unique_lats[i+1] - unique_lats[i] for i in range(len(unique_lats)-1)]
            print(f"CSV latitude spacing: {np.mean(lat_diffs):.6f} ± {np.std(lat_diffs):.6f}")
        
        # Get unique sorted longitudes  
        unique_lons = sorted(csv_sample['x'].unique())[:10]
        if len(unique_lons) > 1:
            lon_diffs = [unique_lons[i+1] - unique_lons[i] for i in range(len(unique_lons)-1)]
            print(f"CSV longitude spacing: {np.mean(lon_diffs):.6f} ± {np.std(lon_diffs):.6f}")
    
    ds.close()
    
    return matches > 0

if __name__ == "__main__":
    validate_with_coordinates()