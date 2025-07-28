#!/usr/bin/env python3
"""Manual CSV export for merged NetCDF dataset."""

import sys
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def export_netcdf_to_csv(nc_file, output_dir=None):
    """Export NetCDF merged dataset to CSV format."""
    nc_path = Path(nc_file)
    if not nc_path.exists():
        print(f"Error: File not found: {nc_file}")
        return
    
    if output_dir is None:
        output_dir = nc_path.parent
    
    print(f"Loading NetCDF file: {nc_path}")
    print(f"File size: {nc_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    try:
        # Load dataset with dask for memory efficiency
        ds = xr.open_dataset(nc_path, chunks='auto')
        print(f"Dataset dimensions: {dict(ds.dims)}")
        print(f"Dataset variables: {list(ds.data_vars)}")
        print(f"Dataset shape: y={ds.dims['y']}, x={ds.dims['x']}")
        
        # Get coordinates
        if 'lat' in ds.coords and 'lon' in ds.coords:
            lats = ds.coords['lat'].values
            lons = ds.coords['lon'].values
            print(f"Coordinate ranges: lat=[{lats.min():.2f}, {lats.max():.2f}], lon=[{lons.min():.2f}, {lons.max():.2f}]")
        else:
            print("Warning: No lat/lon coordinates found")
            lats = None
            lons = None
        
        # Prepare data for CSV export
        print("\nPreparing data for CSV export...")
        
        # Create row for each grid cell
        rows = []
        total_cells = ds.dims['y'] * ds.dims['x']
        
        # Process in chunks to manage memory
        chunk_size = 1000
        processed = 0
        
        for y_idx in range(0, ds.dims['y'], chunk_size):
            y_end = min(y_idx + chunk_size, ds.dims['y'])
            
            # Load chunk
            chunk = ds.isel(y=slice(y_idx, y_end)).compute()
            
            for y in range(y_idx, y_end):
                for x in range(ds.dims['x']):
                    row = {
                        'grid_y': y,
                        'grid_x': x,
                    }
                    
                    # Add coordinates if available
                    if lats is not None and lons is not None:
                        row['latitude'] = float(lats[y])
                        row['longitude'] = float(lons[x])
                    
                    # Add all data variables
                    for var in ds.data_vars:
                        value = chunk[var].isel(y=y-y_idx, x=x).values
                        if isinstance(value, np.ndarray):
                            value = value.item()
                        row[var] = value
                    
                    rows.append(row)
                
                processed = (y - y_idx + 1) * ds.dims['x'] + y_idx * ds.dims['x']
                if processed % 10000 == 0:
                    print(f"Processed {processed}/{total_cells} cells ({processed/total_cells*100:.1f}%)")
        
        print(f"\nTotal rows prepared: {len(rows)}")
        
        # Convert to DataFrame
        print("Converting to DataFrame...")
        df = pd.DataFrame(rows)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = Path(output_dir) / f"merged_data_{timestamp}.csv"
        
        print(f"\nSaving to CSV: {csv_file}")
        df.to_csv(csv_file, index=False)
        
        file_size_mb = csv_file.stat().st_size / 1024 / 1024
        print(f"✓ CSV export complete: {csv_file}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        
        # Show sample
        print("\nSample data (first 5 rows):")
        print(df.head())
        
        # Save compressed version if large
        if file_size_mb > 100:
            print(f"\nFile is large ({file_size_mb:.1f} MB), creating compressed version...")
            csv_gz_file = Path(output_dir) / f"merged_data_{timestamp}.csv.gz"
            df.to_csv(csv_gz_file, index=False, compression='gzip')
            gz_size_mb = csv_gz_file.stat().st_size / 1024 / 1024
            print(f"✓ Compressed CSV: {csv_gz_file} ({gz_size_mb:.2f} MB)")
        
        return str(csv_file)
        
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Find the latest merged dataset
    import glob
    nc_files = sorted(glob.glob("outputs/*/merged_dataset.nc"), 
                     key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    if nc_files:
        latest_nc = nc_files[0]
        print(f"Found {len(nc_files)} merged datasets")
        print(f"Using most recent: {latest_nc}")
        export_netcdf_to_csv(latest_nc)
    else:
        print("No merged datasets found in outputs/*/merged_dataset.nc")
        if len(sys.argv) > 1:
            export_netcdf_to_csv(sys.argv[1])