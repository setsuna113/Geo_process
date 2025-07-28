#!/usr/bin/env python3
"""Debug the complete data flow from TIF → DB → Merge → CSV."""

import numpy as np
import rasterio
from pathlib import Path

from src.database.connection import DatabaseManager
from src.database.schema import DatabaseSchema

def trace_single_coordinate():
    """Trace one coordinate through the entire pipeline."""
    
    # Test coordinate
    test_lat, test_lon = 20.6417, 106.2917
    print(f"=== Tracing coordinate ({test_lat}, {test_lon}) ===\n")
    
    # 1. Check original TIF values
    print("1. ORIGINAL TIF VALUES:")
    tif_files = {
        'terrestrial': '/maps/mwd24/richness/iucn-terrestrial-richness.tif',
        'plants': '/maps/mwd24/richness/daru-plants-richness.tif'
    }
    
    tif_values = {}
    for name, path in tif_files.items():
        try:
            with rasterio.open(path) as src:
                # Get pixel indices for coordinate
                row, col = src.index(test_lon, test_lat)
                
                # Read value
                window = rasterio.windows.Window(col, row, 1, 1)
                data = src.read(1, window=window)
                value = data[0, 0] if data.size > 0 else np.nan
                
                tif_values[name] = value
                print(f"  {name}: value={value}, indices=({row}, {col})")
                print(f"    Bounds: {src.bounds}")
                print(f"    Shape: {src.shape}")
                print(f"    Transform: {src.transform}")
                
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    # 2. Check database passthrough values
    print("\n2. DATABASE PASSTHROUGH VALUES:")
    db = DatabaseManager()
    
    with db.get_cursor() as cursor:
        for name in ['terrestrial', 'plants']:
            table = f'passthrough_{name}_richness'
            
            # First, understand the coordinate system in DB
            cursor.execute(f'''
                SELECT MIN(row_idx) as min_row, MAX(row_idx) as max_row,
                       MIN(col_idx) as min_col, MAX(col_idx) as max_col
                FROM {table}
            ''')
            bounds = cursor.fetchone()
            print(f"\n  {name} table bounds:")
            print(f"    Row range: [{bounds['min_row']}, {bounds['max_row']}]")
            print(f"    Col range: [{bounds['min_col']}, {bounds['max_col']}]")
            
            # Try to find our coordinate
            # Need to figure out the index mapping
            if name == 'terrestrial':
                # From TIF: shape (8280, 21600), bounds (-180, -54.996, 180, 83.004)
                # lat range: -54.996 to 83.004 = 138 degrees over 8280 rows
                # lon range: -180 to 180 = 360 degrees over 21600 cols
                
                lat_min, lat_max = -54.996, 83.004
                lon_min, lon_max = -180, 180
                
                # Calculate indices
                row_idx = int((lat_max - test_lat) / (lat_max - lat_min) * 8280)
                col_idx = int((test_lon - lon_min) / (lon_max - lon_min) * 21600)
                
            else:  # plants
                # From TIF: shape (10410, 21600), bounds (-180, -89.833, 180, 83.667)
                lat_min, lat_max = -89.833, 83.667
                lon_min, lon_max = -180, 180
                
                row_idx = int((lat_max - test_lat) / (lat_max - lat_min) * 10410)
                col_idx = int((test_lon - lon_min) / (lon_max - lon_min) * 21600)
            
            print(f"    Calculated indices for ({test_lat}, {test_lon}): ({row_idx}, {col_idx})")
            
            # Get value from DB
            cursor.execute(f'''
                SELECT value FROM {table}
                WHERE row_idx = %s AND col_idx = %s
            ''', (row_idx, col_idx))
            
            result = cursor.fetchone()
            if result:
                print(f"    DB value: {result['value']}")
            else:
                # Try nearby values
                cursor.execute(f'''
                    SELECT row_idx, col_idx, value 
                    FROM {table}
                    WHERE row_idx BETWEEN %s AND %s
                    AND col_idx BETWEEN %s AND %s
                    ORDER BY ABS(row_idx - %s) + ABS(col_idx - %s)
                    LIMIT 5
                ''', (row_idx-2, row_idx+2, col_idx-2, col_idx+2, row_idx, col_idx))
                
                print(f"    No exact match, nearby values:")
                for row in cursor.fetchall():
                    print(f"      ({row['row_idx']}, {row['col_idx']}): {row['value']}")
    
    # 3. Check how merge stage would read this
    print("\n3. MERGE STAGE LOGIC:")
    print("  The merge stage calls load_passthrough_data() which:")
    print("  - Loads data array from TIF file using rioxarray")
    print("  - Returns raw numpy array without coordinate information")
    print("  - The _align_data_to_common_grid() then tries to map it to common coordinates")
    print("  - This is where the bug likely occurs!")
    
    # 4. Final CSV values (already known)
    print("\n4. FINAL CSV VALUES:")
    print(f"  Terrestrial: 152 (vs TIF: {tif_values.get('terrestrial', 'N/A')})")
    print(f"  Plants: 1656 (vs TIF: {tif_values.get('plants', 'N/A')})")
    
    print("\n=== ANALYSIS ===")
    print("The data flow shows:")
    print("1. TIF files have the correct values")
    print("2. Database stores values with row/col indices")
    print("3. Merge stage loads from TIF (not DB!) but uses wrong coordinate mapping")
    print("4. The 'passthrough' data in DB is effectively ignored during merge!")

if __name__ == "__main__":
    trace_single_coordinate()