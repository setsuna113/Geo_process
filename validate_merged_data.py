#!/usr/bin/env python3
"""Validate merged data by comparing samples from original datasets with CSV output."""

import sys
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from random import sample
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database.connection import DatabaseManager
from src.database.schema import DatabaseSchema

def get_original_samples(n_samples=100):
    """Get random samples from original resampled datasets in database."""
    db = DatabaseManager()
    schema = DatabaseSchema()
    
    print(f"Fetching {n_samples} random samples from database...")
    
    # Query to get random samples from resampling cache
    query = """
    SELECT 
        rc.grid_cell_id,
        gc.x_index,
        gc.y_index,
        gc.center_lat,
        gc.center_lon,
        rc.dataset_name,
        rc.variable_name,
        rc.resampled_value,
        rc.original_mean,
        rc.original_count
    FROM resampling_cache rc
    JOIN grid_cells gc ON rc.grid_cell_id = gc.id
    WHERE rc.resampled_value IS NOT NULL
    AND gc.grid_definition_id = (
        SELECT id FROM grid_definitions 
        WHERE name = 'standard_grid_5km' 
        ORDER BY created_at DESC LIMIT 1
    )
    ORDER BY RANDOM()
    LIMIT %s
    """
    
    with db.get_connection() as conn:
        df_samples = pd.read_sql_query(query, conn, params=(n_samples,))
    
    print(f"Retrieved {len(df_samples)} samples from database")
    return df_samples

def load_merged_csv(csv_path):
    """Load the merged CSV file."""
    print(f"\nLoading merged CSV: {csv_path}")
    
    # Read in chunks for memory efficiency
    chunk_size = 1000000
    chunks = []
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunks.append(chunk)
        if len(chunks) * chunk_size >= 10000000:  # Limit to 10M rows for testing
            print(f"Loaded {len(chunks) * chunk_size:,} rows (limited for testing)")
            break
    
    df_csv = pd.concat(chunks, ignore_index=True)
    print(f"Total rows loaded: {len(df_csv):,}")
    print(f"Columns: {list(df_csv.columns)}")
    
    return df_csv

def validate_samples(df_samples, df_csv):
    """Validate that samples from database exist in CSV with correct values."""
    print("\n=== Validation Results ===")
    
    # Group by dataset for validation
    datasets = df_samples['dataset_name'].unique()
    
    results = {
        'total_samples': len(df_samples),
        'datasets': {},
        'mismatches': []
    }
    
    for dataset in datasets:
        dataset_samples = df_samples[df_samples['dataset_name'] == dataset]
        print(f"\nValidating {dataset}: {len(dataset_samples)} samples")
        
        # Map dataset names to CSV column names
        column_map = {
            'biodiversity_terrestrial_richness_h5': 'terrestrial_richness',
            'biodiversity_plants_richness_h5': 'plants_richness'
        }
        
        csv_column = column_map.get(dataset)
        if not csv_column:
            print(f"  WARNING: Unknown dataset mapping for {dataset}")
            continue
        
        if csv_column not in df_csv.columns:
            print(f"  ERROR: Column {csv_column} not found in CSV")
            continue
        
        found = 0
        value_matches = 0
        
        for _, sample in dataset_samples.iterrows():
            # Find corresponding row in CSV by coordinates
            tolerance = 0.01  # Coordinate tolerance
            
            # Try matching by indices first
            csv_match = df_csv[
                (df_csv['y'] == sample['y_index']) & 
                (df_csv['x'] == sample['x_index'])
            ]
            
            if csv_match.empty:
                # Try matching by coordinates
                csv_match = df_csv[
                    (abs(df_csv['y'] - sample['center_lat']) < tolerance) & 
                    (abs(df_csv['x'] - sample['center_lon']) < tolerance)
                ]
            
            if not csv_match.empty:
                found += 1
                csv_value = csv_match.iloc[0][csv_column]
                db_value = sample['resampled_value']
                
                # Check if values match (with tolerance for floats)
                if pd.isna(csv_value) and pd.isna(db_value):
                    value_matches += 1
                elif not pd.isna(csv_value) and not pd.isna(db_value):
                    if abs(csv_value - db_value) < 0.0001:
                        value_matches += 1
                    else:
                        mismatch = {
                            'dataset': dataset,
                            'location': f"({sample['center_lat']:.2f}, {sample['center_lon']:.2f})",
                            'indices': f"({sample['y_index']}, {sample['x_index']})",
                            'db_value': db_value,
                            'csv_value': csv_value,
                            'difference': abs(csv_value - db_value)
                        }
                        results['mismatches'].append(mismatch)
                        if len(results['mismatches']) <= 5:  # Show first 5 mismatches
                            print(f"  MISMATCH at {mismatch['location']}: "
                                  f"DB={db_value:.4f}, CSV={csv_value:.4f}, "
                                  f"diff={mismatch['difference']:.6f}")
        
        results['datasets'][dataset] = {
            'samples': len(dataset_samples),
            'found': found,
            'value_matches': value_matches,
            'match_rate': value_matches / len(dataset_samples) * 100 if len(dataset_samples) > 0 else 0
        }
        
        print(f"  Found in CSV: {found}/{len(dataset_samples)} ({found/len(dataset_samples)*100:.1f}%)")
        print(f"  Value matches: {value_matches}/{found} ({value_matches/found*100:.1f}% of found)")
    
    return results

def check_coordinate_system(df_csv):
    """Check if CSV uses indices or lat/lon coordinates."""
    print("\n=== Coordinate System Check ===")
    
    # Check x,y ranges
    print(f"X range: [{df_csv['x'].min():.2f}, {df_csv['x'].max():.2f}]")
    print(f"Y range: [{df_csv['y'].min():.2f}, {df_csv['y'].max():.2f}]")
    
    # Determine coordinate system
    if df_csv['x'].max() > 360 or df_csv['y'].max() > 180:
        print("Appears to use grid indices (not lat/lon)")
        return 'indices'
    else:
        print("Appears to use lat/lon coordinates")
        return 'latlon'

def main():
    # Find latest CSV
    csv_files = sorted(Path("outputs").glob("*/merged_data*_valid_only.csv"), 
                      key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not csv_files:
        print("No merged CSV files found")
        return
    
    csv_path = csv_files[0]
    print(f"Using CSV: {csv_path}")
    print(f"File size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Get samples from database
    df_samples = get_original_samples(n_samples=200)
    
    # Load CSV (limited for testing)
    df_csv = load_merged_csv(csv_path)
    
    # Check coordinate system
    coord_system = check_coordinate_system(df_csv)
    
    # Validate samples
    results = validate_samples(df_samples, df_csv)
    
    # Summary
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Total samples tested: {results['total_samples']}")
    
    overall_found = 0
    overall_matches = 0
    
    for dataset, stats in results['datasets'].items():
        print(f"\n{dataset}:")
        print(f"  Samples found: {stats['found']}/{stats['samples']} ({stats['found']/stats['samples']*100:.1f}%)")
        print(f"  Values correct: {stats['value_matches']}/{stats['found']} ({stats['match_rate']:.1f}%)")
        overall_found += stats['found']
        overall_matches += stats['value_matches']
    
    print(f"\nOVERALL:")
    print(f"  Found rate: {overall_found}/{results['total_samples']} ({overall_found/results['total_samples']*100:.1f}%)")
    if overall_found > 0:
        print(f"  Accuracy: {overall_matches}/{overall_found} ({overall_matches/overall_found*100:.1f}%)")
    
    if results['mismatches']:
        print(f"\nFound {len(results['mismatches'])} value mismatches")
        
    # Save detailed results
    results_file = csv_path.parent / "validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main()