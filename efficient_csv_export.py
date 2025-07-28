#!/usr/bin/env python3
"""Efficient CSV export that skips NaN values and uses proper xarray methods."""

import sys
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def export_netcdf_to_csv_efficient(nc_file, output_dir=None, skip_nan=True, sample_rate=1):
    """Export NetCDF to CSV, optionally skipping NaN values and sampling."""
    nc_path = Path(nc_file)
    if not nc_path.exists():
        print(f"Error: File not found: {nc_file}")
        return
    
    if output_dir is None:
        output_dir = nc_path.parent
    
    print(f"Loading NetCDF file: {nc_path}")
    print(f"File size: {nc_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    try:
        # Load dataset
        ds = xr.open_dataset(nc_path)
        print(f"Dataset dimensions: {dict(ds.sizes)}")  # Use sizes instead of dims
        print(f"Dataset variables: {list(ds.data_vars)}")
        print(f"Total cells: {ds.sizes['y'] * ds.sizes['x']:,}")
        
        # Convert to dataframe more efficiently
        print("\nConverting to DataFrame...")
        print(f"Skip NaN: {skip_nan}, Sample rate: {sample_rate}")
        
        if sample_rate > 1:
            # Sample the data
            ds_sampled = ds.isel(
                y=slice(0, ds.sizes['y'], sample_rate),
                x=slice(0, ds.sizes['x'], sample_rate)
            )
            print(f"Sampled to: {ds_sampled.sizes['y']} x {ds_sampled.sizes['x']} = {ds_sampled.sizes['y'] * ds_sampled.sizes['x']:,} cells")
            df = ds_sampled.to_dataframe()
        else:
            # Full dataset
            df = ds.to_dataframe()
        
        # Reset index to get coordinates as columns
        df = df.reset_index()
        
        # Count non-NaN values
        if skip_nan:
            print("\nChecking for non-NaN values...")
            # Check which rows have at least one non-NaN data value
            data_vars = list(ds.data_vars)
            mask = df[data_vars].notna().any(axis=1)
            valid_count = mask.sum()
            print(f"Rows with valid data: {valid_count:,} out of {len(df):,} ({valid_count/len(df)*100:.1f}%)")
            
            if skip_nan:
                df = df[mask]
                print(f"Filtered to {len(df):,} rows with valid data")
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_sample{sample_rate}" if sample_rate > 1 else ""
        suffix += "_valid_only" if skip_nan else "_all"
        csv_file = Path(output_dir) / f"merged_data_{timestamp}{suffix}.csv"
        
        print(f"\nSaving to CSV: {csv_file}")
        df.to_csv(csv_file, index=False)
        
        file_size_mb = csv_file.stat().st_size / 1024 / 1024
        print(f"\n✓ CSV export complete!")
        print(f"  File: {csv_file}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        
        # Show sample
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nColumn summary:")
        for col in df.columns:
            if col in data_vars:
                non_nan = df[col].notna().sum()
                print(f"  {col}: {non_nan:,} non-NaN values ({non_nan/len(df)*100:.1f}%)")
        
        # Compress if large
        if file_size_mb > 50:
            csv_gz = csv_file.with_suffix('.csv.gz')
            print(f"\nCompressing to {csv_gz.name}...")
            df.to_csv(csv_gz, index=False, compression='gzip')
            gz_size = csv_gz.stat().st_size / 1024 / 1024
            print(f"✓ Compressed size: {gz_size:.2f} MB ({gz_size/file_size_mb*100:.1f}% of original)")
        
        ds.close()
        return str(csv_file)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import glob
    
    # Parse arguments
    skip_nan = "--all" not in sys.argv
    sample_rate = 1
    
    for arg in sys.argv[1:]:
        if arg.startswith("--sample="):
            sample_rate = int(arg.split("=")[1])
    
    # Find latest file
    nc_files = sorted(glob.glob("outputs/*/merged_dataset.nc"), 
                     key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    if nc_files:
        print(f"Found {len(nc_files)} merged datasets")
        print(f"Using most recent: {nc_files[0]}")
        print(f"Options: skip_nan={skip_nan}, sample_rate={sample_rate}")
        export_netcdf_to_csv_efficient(nc_files[0], skip_nan=skip_nan, sample_rate=sample_rate)
    else:
        print("No merged datasets found")