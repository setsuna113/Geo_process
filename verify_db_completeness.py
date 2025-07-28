#!/usr/bin/env python3
"""Verify DB contains all data and check for missing entries."""

import numpy as np
from pathlib import Path
from src.database.connection import DatabaseManager

def verify_db_completeness():
    """Check if DB has all the data from both datasets."""
    
    print("=== VERIFYING DATABASE COMPLETENESS ===\n")
    
    db = DatabaseManager()
    
    # 1. Check passthrough table statistics
    print("1. DATABASE PASSTHROUGH TABLES:")
    with db.get_cursor() as cursor:
        for dataset in ['terrestrial', 'plants']:
            table = f'passthrough_{dataset}_richness'
            
            # Get basic stats
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(DISTINCT row_idx) as unique_rows,
                    COUNT(DISTINCT col_idx) as unique_cols,
                    MIN(row_idx) as min_row, MAX(row_idx) as max_row,
                    MIN(col_idx) as min_col, MAX(col_idx) as max_col,
                    MIN(value) as min_val, MAX(value) as max_val,
                    AVG(value) as avg_val
                FROM {table}
            """)
            
            stats = cursor.fetchone()
            print(f"\n   {dataset}_richness:")
            print(f"     Total entries: {stats['total_entries']:,}")
            print(f"     Grid dimensions: {stats['unique_rows']} x {stats['unique_cols']}")
            print(f"     Row range: [{stats['min_row']}, {stats['max_row']}]")
            print(f"     Col range: [{stats['min_col']}, {stats['max_col']}]")
            print(f"     Value range: [{stats['min_val']:.0f}, {stats['max_val']:.0f}]")
            print(f"     Average: {stats['avg_val']:.2f}")
            
            # Expected entries
            expected = (stats['max_row'] - stats['min_row'] + 1) * (stats['max_col'] - stats['min_col'] + 1)
            coverage = (stats['total_entries'] / expected) * 100
            print(f"     Coverage: {stats['total_entries']:,} / {expected:,} = {coverage:.1f}%")
    
    # 2. Check for data outside merged bounds
    print("\n2. CHECKING FOR DATA OUTSIDE MERGED BOUNDS:")
    
    # From our analysis: merged uses plants bounds as common bounds
    # Plants: 10410 x 21600
    # Terrestrial: 8280 x 21600
    
    with db.get_cursor() as cursor:
        # Check terrestrial data that would be outside plants bounds
        cursor.execute("""
            SELECT COUNT(*) as outside_count,
                   MIN(row_idx) as min_outside_row,
                   MAX(row_idx) as max_outside_row
            FROM passthrough_terrestrial_richness
            WHERE row_idx >= 8280  -- Terrestrial only goes to 8279
        """)
        
        result = cursor.fetchone()
        print(f"\n   Terrestrial data outside its bounds: {result['outside_count']} entries")
        
        # Check if terrestrial has full coverage within its bounds
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT row_idx || ',' || col_idx) as cells_with_data,
                8280 * 21600 as total_possible_cells
            FROM passthrough_terrestrial_richness
            WHERE row_idx < 8280
        """)
        
        coverage = cursor.fetchone()
        print(f"   Terrestrial coverage: {coverage['cells_with_data']:,} / {coverage['total_possible_cells']:,}")
    
    # 3. Sample some edge cases
    print("\n3. EDGE CASE SAMPLES:")
    
    with db.get_cursor() as cursor:
        # Check corners of terrestrial data
        print("\n   Terrestrial corners:")
        corners = [(0, 0), (0, 21599), (8279, 0), (8279, 21599)]
        
        for row, col in corners:
            cursor.execute("""
                SELECT value FROM passthrough_terrestrial_richness
                WHERE row_idx = %s AND col_idx = %s
            """, (row, col))
            
            result = cursor.fetchone()
            val = result['value'] if result else 'NO DATA'
            print(f"     ({row}, {col}): {val}")
        
        # Check where terrestrial stops (bottom edge)
        print("\n   Terrestrial bottom edge (where it differs from plants):")
        cursor.execute("""
            SELECT col_idx, value 
            FROM passthrough_terrestrial_richness
            WHERE row_idx = 8279
            AND col_idx IN (0, 5400, 10800, 16200, 21599)
            ORDER BY col_idx
        """)
        
        for row in cursor.fetchall():
            print(f"     Row 8279, Col {row['col_idx']}: {row['value']}")
    
    # 4. Check resampled_datasets metadata
    print("\n4. RESAMPLED DATASETS METADATA:")
    
    with db.get_cursor() as cursor:
        cursor.execute("""
            SELECT name, shape_height, shape_width, bounds, metadata
            FROM resampled_datasets
        """)
        
        for row in cursor.fetchall():
            print(f"\n   {row['name']}:")
            print(f"     Shape: {row['shape_height']} x {row['shape_width']}")
            print(f"     Bounds: {row['bounds']}")
            if row['metadata']:
                print(f"     Passthrough: {row['metadata'].get('passthrough', False)}")

if __name__ == "__main__":
    verify_db_completeness()