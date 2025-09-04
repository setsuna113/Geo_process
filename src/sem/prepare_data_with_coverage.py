#!/usr/bin/env python3

"""
Prepare biodiversity data with MOD44B vegetation cover for SEM analysis.
This script is designed to be reusable with configurable parameters.
"""

import pandas as pd
import pyarrow.feather as feather
import numpy as np
import argparse
import os
import sys
from pathlib import Path

def load_data(input_file):
    """Load data from Arrow or CSV file."""
    file_ext = Path(input_file).suffix.lower()
    
    if file_ext == '.arrow' or file_ext == '.feather':
        print(f"Loading Arrow file: {input_file}")
        return feather.read_feather(input_file)
    elif file_ext == '.csv':
        print(f"Loading CSV file: {input_file}")
        return pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def calculate_mycorrhizal_richness(df, am_col='am_fungi_richness', ecm_col='ecm_fungi_richness'):
    """Calculate combined mycorrhizal richness (vectorized for speed)."""
    df['mycorrhizal_richness'] = np.where(
        df[am_col].notna() & df[ecm_col].notna(),
        df[am_col] + df[ecm_col],
        np.where(
            df[am_col].notna(),
            df[am_col],
            np.where(
                df[ecm_col].notna(),
                df[ecm_col],
                np.nan
            )
        )
    )
    return df

def prepare_biodiversity_data(
    input_file,
    output_file,
    sample_size=None,
    random_state=42,
    required_columns=None,
    column_mapping=None,
    stratify_by=None
):
    """
    Prepare biodiversity data for SEM analysis.
    
    Parameters:
    -----------
    input_file : str
        Path to input data file (Arrow or CSV)
    output_file : str
        Path to output CSV file
    sample_size : int, optional
        Number of rows to sample. If None, use all data
    random_state : int
        Random seed for reproducibility
    required_columns : list, optional
        Columns that must be non-null for a row to be included
    column_mapping : dict, optional
        Dictionary to rename columns
    stratify_by : str, optional
        Column name for stratified sampling (e.g., 'biome')
    
    Returns:
    --------
    pd.DataFrame : Prepared data
    """
    
    # Load data
    df = load_data(input_file)
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
    
    # Apply column mapping if provided
    if column_mapping is None:
        # Default mapping for biodiversity data
        column_mapping = {
            'plants_richness': 'plant_richness',
            'terrestrial_richness': 'animal_richness',
            'annual_temp': 'bio01',
            'temp_seasonality': 'bio04', 
            'annual_precip': 'bio12',
            'precip_seasonality': 'bio15',
            'pet': 'pet_mean',
            'modis_tree_cover_pct': 'tree_cover',
            'modis_nontree_veg_pct': 'nontree_veg',
            'modis_bare_ground_pct': 'bare_ground'
        }
    
    # Only rename columns that exist
    existing_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_mappings)
    print(f"Renamed {len(existing_mappings)} columns")
    
    # Add derived variables if needed
    if 'pet_range' not in df.columns and 'bio04' in df.columns:
        df['pet_range'] = df['bio04'] * 0.1  # Placeholder calculation
    
    # Calculate mycorrhizal richness if component columns exist
    if 'am_fungi_richness' in df.columns and 'ecm_fungi_richness' in df.columns:
        df = calculate_mycorrhizal_richness(df)
        print("Calculated combined mycorrhizal richness")
    
    # Set default required columns if not provided
    if required_columns is None:
        required_columns = [
            'mycorrhizal_richness', 'plant_richness', 'animal_richness',
            'bio01', 'bio12', 'elevation'
        ]
        # Add coverage columns if they exist
        if 'tree_cover' in df.columns:
            required_columns.extend(['tree_cover', 'nontree_veg'])
    
    # Filter to columns that actually exist
    required_columns = [col for col in required_columns if col in df.columns]
    
    # Remove rows with missing critical data
    print(f"\nRemoving rows with missing data in: {required_columns}")
    initial_rows = len(df)
    df_clean = df.dropna(subset=required_columns)
    print(f"Removed {initial_rows - len(df_clean)} rows with missing data")
    print(f"Clean data shape: {df_clean.shape}")
    
    # Sample if requested
    if sample_size is not None and len(df_clean) > sample_size:
        print(f"\nSampling {sample_size} rows...")
        
        if stratify_by and stratify_by in df_clean.columns:
            # Stratified sampling
            print(f"Using stratified sampling by {stratify_by}")
            df_sample = df_clean.groupby(stratify_by, group_keys=False).apply(
                lambda x: x.sample(int(np.rint(sample_size * len(x) / len(df_clean))), 
                                   random_state=random_state)
            )
        else:
            # Simple random sampling
            df_sample = df_clean.sample(n=sample_size, random_state=random_state)
    else:
        df_sample = df_clean
    
    print(f"Final sample shape: {df_sample.shape}")
    
    # Save to output file
    output_ext = Path(output_file).suffix.lower()
    if output_ext == '.csv':
        df_sample.to_csv(output_file, index=False)
    elif output_ext in ['.arrow', '.feather']:
        feather.write_feather(df_sample, output_file)
    else:
        raise ValueError(f"Unsupported output format: {output_ext}")
    
    print(f"\nSaved to {output_file}")
    
    # Print summary statistics
    print("\nData summary:")
    summary_cols = ['mycorrhizal_richness', 'plant_richness', 'animal_richness']
    if 'tree_cover' in df_sample.columns:
        summary_cols.extend(['tree_cover', 'nontree_veg'])
    
    summary_cols = [col for col in summary_cols if col in df_sample.columns]
    print(df_sample[summary_cols].describe())
    
    # Print vegetation cover correlation if available
    if 'tree_cover' in df_sample.columns and 'nontree_veg' in df_sample.columns:
        corr = df_sample[['tree_cover', 'nontree_veg']].corr().iloc[0, 1]
        print(f"\nTree vs non-tree vegetation correlation: r = {corr:.3f}")
    
    return df_sample

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Prepare biodiversity data for SEM analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python prepare_data_with_coverage.py input.arrow output.csv
  
  # Sample 1 million rows
  python prepare_data_with_coverage.py input.arrow output.csv --sample 1000000
  
  # Stratified sampling by biome
  python prepare_data_with_coverage.py input.arrow output.csv --sample 500000 --stratify biome
  
  # Specify required columns
  python prepare_data_with_coverage.py input.arrow output.csv --required mycorrhizal_richness plant_richness
        """
    )
    
    parser.add_argument('input_file', help='Input data file (Arrow or CSV)')
    parser.add_argument('output_file', help='Output file (CSV or Arrow)')
    parser.add_argument('--sample', type=int, help='Sample size (default: use all data)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--stratify', help='Column for stratified sampling')
    parser.add_argument('--required', nargs='+', help='Required non-null columns')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Prepare data
    try:
        prepare_biodiversity_data(
            input_file=args.input_file,
            output_file=args.output_file,
            sample_size=args.sample,
            random_state=args.seed,
            required_columns=args.required,
            stratify_by=args.stratify
        )
        print("\nData preparation complete!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # If run without arguments, use defaults for backward compatibility
    if len(sys.argv) == 1:
        print("Running with default settings for backward compatibility...")
        prepare_biodiversity_data(
            input_file="/scratch/yl998/biodiversity_climate_production_WITH_MOD44B.arrow",
            output_file="/scratch/yl998/biodiv_with_coverage_sample.csv",
            sample_size=500000,
            random_state=42
        )
    else:
        main()