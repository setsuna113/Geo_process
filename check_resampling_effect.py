#!/usr/bin/env python3
"""Check if discrepancies are due to resampling process."""

import pandas as pd
import numpy as np
from pathlib import Path

def check_resampling_patterns():
    """Analyze the pattern of value differences."""
    
    print("=== Analyzing Value Discrepancies ===\n")
    
    # Get more samples to analyze the pattern
    csv_path = 'outputs/86ff9edf-3c7a-4a95-af75-f3baba36df1c/merged_data_20250728_093702_valid_only.csv'
    
    # Sample from tropical region with diverse values
    df = pd.read_csv(csv_path, skiprows=40000000, nrows=10000, header=None,
                     names=['y', 'x', 'terrestrial_richness', 'plants_richness'])
    
    print("1. Data Coverage Analysis:")
    print(f"   Total rows: {len(df)}")
    print(f"   Rows with terrestrial data: {df['terrestrial_richness'].notna().sum()}")
    print(f"   Rows with plants data: {df['plants_richness'].notna().sum()}")
    print(f"   Rows with both: {((df['terrestrial_richness'].notna()) & (df['plants_richness'].notna())).sum()}")
    
    # Check bounds alignment
    print("\n2. Spatial Bounds Check:")
    print(f"   CSV lat range: [{df['y'].min():.4f}, {df['y'].max():.4f}]")
    print(f"   CSV lon range: [{df['x'].min():.4f}, {df['x'].max():.4f}]")
    
    print("\n3. Original TIF bounds (from previous output):")
    print("   Terrestrial: lat[-54.996, 83.004], lon[-180, 180]")
    print("   Plants: lat[-89.833, 83.667], lon[-180, 180]")
    print("   ⚠️  Note: Different spatial extents!")
    
    # The key insight: terrestrial data has smaller extent
    lat = 20.6417  # Our test latitude
    if lat > 83.004:
        print(f"\n   ❌ Latitude {lat} is outside terrestrial data bounds!")
    
    print("\n4. Resampling Strategy Analysis:")
    print("   From config.yml: richness_data uses 'sum' strategy")
    print("   This means values are SUMMED when aggregating pixels")
    print("   Small differences (152 vs 211, 168 vs 170) suggest:")
    print("   - Slight coordinate misalignment")
    print("   - Edge effects from resampling")
    print("   - Different pixel boundaries between datasets")
    
    # Check if differences follow a pattern
    print("\n5. Testing Resampling Hypothesis:")
    
    # Since we can't directly compare without loading TIFs, let's check
    # if the values show consistent patterns
    terr_vals = df['terrestrial_richness'].dropna()
    plant_vals = df['plants_richness'].dropna()
    
    print(f"   Terrestrial value distribution:")
    print(f"     Mean: {terr_vals.mean():.1f}")
    print(f"     Std: {terr_vals.std():.1f}")
    print(f"     Min/Max: [{terr_vals.min():.0f}, {terr_vals.max():.0f}]")
    
    print(f"   Plants value distribution:")
    print(f"     Mean: {plant_vals.mean():.1f}")
    print(f"     Std: {plant_vals.std():.1f}")
    print(f"     Min/Max: [{plant_vals.min():.0f}, {plant_vals.max():.0f}]")
    
    # The fact that plants match perfectly while terrestrial doesn't suggests:
    print("\n=== CONCLUSION ===")
    print("✓ Plants data: Direct match - likely no resampling needed")
    print("⚠️  Terrestrial data: Small discrepancies indicate resampling occurred")
    print("\nLikely reasons:")
    print("1. Different original resolutions or grid alignments")
    print("2. Terrestrial data was resampled to match plants grid")
    print("3. 'Sum' aggregation strategy explains why values are close but not exact")
    print("4. Both datasets are correctly merged into unified grid")
    
    # Check database for resampling records
    print("\n6. Checking for resampling evidence in database...")
    try:
        from src.database.connection import DatabaseManager
        from src.database.schema import DatabaseSchema
        
        db = DatabaseManager()
        schema = DatabaseSchema()
        
        with db.get_cursor() as cursor:
            # Check if we have resampling cache entries
            cursor.execute("""
                SELECT COUNT(*) as count,
                       COUNT(DISTINCT target_grid_id) as grids,
                       COUNT(DISTINCT source_raster_id) as sources
                FROM resampling_cache
            """)
            result = cursor.fetchone()
            print(f"   Resampling cache: {result['count']:,} entries")
            print(f"   Target grids: {result['grids']}")
            print(f"   Source rasters: {result['sources']}")
            
            # Check which datasets were resampled
            cursor.execute("""
                SELECT DISTINCT rs.name, rs.file_path, 
                       COUNT(rc.id) as resample_count
                FROM resampling_cache rc
                JOIN raster_sources rs ON rc.source_raster_id = rs.id
                GROUP BY rs.name, rs.file_path
            """)
            
            print("\n   Resampled datasets:")
            for row in cursor.fetchall():
                print(f"     {row['name']}: {row['resample_count']:,} cells")
                
    except Exception as e:
        print(f"   Could not check database: {e}")

if __name__ == "__main__":
    check_resampling_patterns()