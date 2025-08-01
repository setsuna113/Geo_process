#!/usr/bin/env python3
"""Inspect parquet file to verify its contents and structure."""

import pandas as pd
import numpy as np
import sys

def inspect_parquet(file_path):
    """Inspect a parquet file and display comprehensive information."""
    print(f"Inspecting parquet file: {file_path}")
    print("=" * 80)
    
    # Read the parquet file
    df = pd.read_parquet(file_path)
    
    # Basic information
    print("\n1. BASIC INFORMATION:")
    print(f"   - Shape: {df.shape} (rows: {df.shape[0]:,}, columns: {df.shape[1]})")
    print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column information
    print("\n2. COLUMN INFORMATION:")
    print("   Column Name                 | Data Type | Non-Null Count | Null Count | Null %")
    print("   " + "-" * 76)
    for col in df.columns:
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"   {col:<27} | {str(df[col].dtype):<9} | {non_null:>13,} | {null_count:>10,} | {null_pct:>5.1f}%")
    
    # Sample data
    print("\n3. SAMPLE DATA (first 5 rows):")
    print(df.head())
    
    print("\n4. SAMPLE DATA (random 5 rows):")
    if len(df) > 5:
        print(df.sample(n=min(5, len(df)), random_state=42))
    
    # Statistical summary for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print("\n5. STATISTICAL SUMMARY (numeric columns):")
        print(df[numeric_cols].describe())
        
        # Check for specific biodiversity columns
        biodiversity_cols = ['plants_richness', 'terrestrial_richness', 
                           'am_fungi_richness', 'ecm_fungi_richness']
        found_bio_cols = [col for col in biodiversity_cols if col in df.columns]
        
        if found_bio_cols:
            print("\n6. BIODIVERSITY DATA SUMMARY:")
            for col in found_bio_cols:
                print(f"\n   {col}:")
                print(f"   - Min: {df[col].min():.4f}")
                print(f"   - Max: {df[col].max():.4f}")
                print(f"   - Mean: {df[col].mean():.4f}")
                print(f"   - Std: {df[col].std():.4f}")
                print(f"   - Non-zero values: {(df[col] > 0).sum():,} ({(df[col] > 0).sum() / len(df) * 100:.1f}%)")
    
    # Check for coordinate columns
    coord_cols = ['x', 'y', 'longitude', 'latitude', 'lon', 'lat']
    found_coord_cols = [col for col in coord_cols if col in df.columns]
    
    if found_coord_cols:
        print("\n7. SPATIAL EXTENT:")
        for col in found_coord_cols:
            if col in ['x', 'longitude', 'lon']:
                print(f"   {col}: [{df[col].min():.6f}, {df[col].max():.6f}]")
            elif col in ['y', 'latitude', 'lat']:
                print(f"   {col}: [{df[col].min():.6f}, {df[col].max():.6f}]")
    
    # Check data quality
    print("\n8. DATA QUALITY CHECKS:")
    print(f"   - Total NaN values: {df.isna().sum().sum():,}")
    print(f"   - Rows with any NaN: {df.isna().any(axis=1).sum():,} ({df.isna().any(axis=1).sum() / len(df) * 100:.1f}%)")
    print(f"   - Completely empty rows: {df.isna().all(axis=1).sum():,}")
    
    # Check for duplicates based on coordinates if they exist
    if 'x' in df.columns and 'y' in df.columns:
        duplicates = df.duplicated(subset=['x', 'y'], keep=False).sum()
        print(f"   - Duplicate coordinates: {duplicates:,} ({duplicates / len(df) * 100:.1f}%)")

if __name__ == "__main__":
    parquet_file = "/home/yl998/dev/geo/outputs/45d78409-3818-4eb6-960f-c00216110460/merged_data_45d78409-3818-4eb6-960f-c00216110460_20250801_001157.parquet"
    inspect_parquet(parquet_file)