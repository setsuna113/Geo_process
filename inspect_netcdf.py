#!/usr/bin/env python3
"""Quick inspection of NetCDF file structure."""

import xarray as xr
from pathlib import Path

# Find latest merged file
nc_files = sorted(Path("outputs").glob("*/merged_dataset.nc"), 
                 key=lambda p: p.stat().st_mtime, reverse=True)

if nc_files:
    nc_file = nc_files[0]
    print(f"Inspecting: {nc_file}")
    print(f"File size: {nc_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Open and inspect
    ds = xr.open_dataset(nc_file)
    print(f"\nDimensions: {dict(ds.sizes)}")
    print(f"Coordinates: {list(ds.coords)}")
    print(f"Data variables: {list(ds.data_vars)}")
    
    # Check a data variable
    if ds.data_vars:
        var = list(ds.data_vars)[0]
        print(f"\nFirst variable '{var}':")
        print(f"  Shape: {ds[var].shape}")
        print(f"  Dtype: {ds[var].dtype}")
        print(f"  Memory if loaded: {ds[var].nbytes / 1024 / 1024:.2f} MB")
        
        # Check for valid data
        sample = ds[var].isel(y=0, x=0).compute()
        print(f"  Sample value at (0,0): {sample.values}")
    
    ds.close()
else:
    print("No merged datasets found")