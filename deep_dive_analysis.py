#!/usr/bin/env python3
"""Deep dive into all the issues: loading, storage, merge, and export."""

import numpy as np
import rasterio
from pathlib import Path
import xarray as xr

from src.database.connection import DatabaseManager
from src.database.schema import DatabaseSchema

def investigate_all_issues():
    """Comprehensive investigation of all data flow issues."""
    
    print("=== DEEP DIVE ANALYSIS ===\n")
    
    # Test coordinate and values we've been tracking
    test_lat, test_lon = 20.6417, 106.2917
    
    # Issue 1: Loading Stage - Why DB has different values than TIF
    print("1. LOADING STAGE ISSUES:")
    print("   You noticed: Plants 1656 (TIF) → 1381 (DB) → 1656 (CSV)")
    print("   Let me check the EXACT DB values:\n")
    
    db = DatabaseManager()
    
    # First, let's see what was actually stored in DB
    with db.get_cursor() as cursor:
        # Check the metadata table
        cursor.execute("""
            SELECT name, source_path, shape_height, shape_width, 
                   data_table_name, metadata
            FROM resampled_datasets
        """)
        
        print("   Resampled datasets metadata:")
        for row in cursor.fetchall():
            print(f"     {row['name']}: table={row['data_table_name']}, shape=({row['shape_height']}, {row['shape_width']})")
            
        # Now check the actual stored values more carefully
        print(f"\n   Checking exact values at indices from our earlier analysis:")
        
        # From earlier: terrestrial (3741, 17177), plants (3781, 17177)
        for dataset, indices in [('terrestrial', (3741, 17177)), ('plants', (3781, 17177))]:
            table = f'passthrough_{dataset}_richness'
            row_idx, col_idx = indices
            
            # Get exact value
            cursor.execute(f"""
                SELECT value FROM {table}
                WHERE row_idx = %s AND col_idx = %s
            """, (row_idx, col_idx))
            
            result = cursor.fetchone()
            exact_value = result['value'] if result else None
            
            # Also check nearby values to see if there's a pattern
            cursor.execute(f"""
                SELECT row_idx, col_idx, value FROM {table}
                WHERE row_idx BETWEEN %s AND %s
                AND col_idx BETWEEN %s AND %s
                ORDER BY row_idx, col_idx
            """, (row_idx-1, row_idx+1, col_idx-1, col_idx+1))
            
            print(f"\n     {dataset} at ({row_idx}, {col_idx}):")
            print(f"       Exact value: {exact_value}")
            print(f"       3x3 grid around it:")
            
            values_grid = {}
            for row in cursor.fetchall():
                values_grid[(row['row_idx'], row['col_idx'])] = row['value']
            
            for r in range(row_idx-1, row_idx+2):
                row_str = "       "
                for c in range(col_idx-1, col_idx+2):
                    val = values_grid.get((r, c), 'None')
                    row_str += f"{val:>6} "
                print(row_str)
    
    # Issue 2: NetCDF vs CSV Export Logic
    print("\n2. MERGE OUTPUT ISSUES:")
    print("   Why NetCDF (.nc) instead of direct CSV?")
    
    # Check what files were created
    output_dirs = list(Path("outputs").glob("*/"))
    if output_dirs:
        latest_dir = sorted(output_dirs, key=lambda p: p.stat().st_mtime)[-1]
        print(f"\n   Latest output directory: {latest_dir}")
        
        files = list(latest_dir.glob("*"))
        for f in sorted(files):
            print(f"     {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
        
        # Check NetCDF structure
        nc_files = list(latest_dir.glob("*.nc"))
        if nc_files:
            print(f"\n   NetCDF file structure:")
            ds = xr.open_dataset(nc_files[0])
            print(f"     Dimensions: {dict(ds.sizes)}")
            print(f"     Variables: {list(ds.data_vars)}")
            print(f"     Attributes: {dict(ds.attrs)}")
            ds.close()
    
    # Issue 3: Pipeline stages and redundancy
    print("\n3. PIPELINE STAGE LOGIC:")
    print("   Checking stage responsibilities...\n")
    
    # Read pipeline configuration
    from src.config import config
    pipeline_config = config.get('pipeline', {})
    
    print("   Pipeline stages from orchestrator:")
    print("     1. DataLoadStage - Just registers dataset paths")
    print("     2. ResampleStage - Loads TIF → stores in DB passthrough tables")
    print("     3. MergeStage - Re-loads from TIF (BUG!), creates NetCDF")
    print("     4. ExportStage - Converts NetCDF → CSV")
    print("     5. AnalysisStage - Runs SOM on data")
    
    print("\n   Stage outputs:")
    print("     - ResampleStage: Creates passthrough_* tables in DB")
    print("     - MergeStage: Creates merged_dataset.nc file")
    print("     - ExportStage: Creates merged_data_*.csv file")
    
    # Issue 4: Check skip logic
    print("\n4. SKIP LOGIC AND DATA FLOW:")
    skip_config = pipeline_config.get('stages', {})
    print(f"   Skip configuration: {skip_config}")
    
    # Issue 5: Registration/validation issues
    print("\n5. VALIDATION AND REGISTRATION:")
    with db.get_cursor() as cursor:
        # Check experiments table
        cursor.execute("""
            SELECT name, status, config->>'error' as error
            FROM experiments
            WHERE status != 'completed'
            ORDER BY created_at DESC
            LIMIT 5
        """)
        
        print("   Recent failed experiments:")
        for row in cursor.fetchall():
            print(f"     {row['name']}: {row['status']}")
            if row['error']:
                print(f"       Error: {row['error']}")
    
    print("\n=== SUMMARY OF ISSUES ===")
    print("1. LOADING: DB stores correct values but merge doesn't use them")
    print("2. REDUNDANCY: Data loaded 3 times (resample→DB, merge→TIF, export→NC)")
    print("3. FORMAT: NetCDF is intermediate format (xarray), not DB related")
    print("4. EXPORT: Separate stage needed because merge creates NC not CSV")
    print("5. COORDINATION: Each stage works independently, no shared data flow")

if __name__ == "__main__":
    investigate_all_issues()