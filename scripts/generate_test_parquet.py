#!/usr/bin/env python3
"""Generate test parquet data for ML pipeline testing."""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def generate_test_biodiversity_data(n_samples=1000, output_path='outputs/test_biodiversity.parquet'):
    """Generate synthetic biodiversity data for testing ML pipeline."""
    
    np.random.seed(42)
    
    # Generate spatial coordinates
    lat = np.random.uniform(-60, 60, n_samples)
    lon = np.random.uniform(-180, 180, n_samples)
    
    # Generate spatial features
    data = {
        'latitude': lat,
        'longitude': lon,
        'grid_id': [f'cell_{i}' for i in range(n_samples)],
        
        # Richness data (target variables)
        'plants_richness': np.random.poisson(50, n_samples) + np.abs(lat) * 0.5,  # Higher at poles
        'animals_richness': np.random.poisson(30, n_samples) + (60 - np.abs(lat)) * 0.3,  # Higher at equator
        'total_richness': None,  # Will calculate
        
        # Environmental features
        'temperature': 30 - np.abs(lat) * 0.5 + np.random.normal(0, 2, n_samples),
        'precipitation': 1000 + np.random.normal(0, 200, n_samples),
        'elevation': np.random.exponential(500, n_samples),
        
        # Some missing values for testing imputation
        'soil_ph': np.where(np.random.random(n_samples) > 0.9, np.nan, 
                           np.random.uniform(4, 8, n_samples)),
        'forest_cover': np.where(np.random.random(n_samples) > 0.95, np.nan,
                               np.random.uniform(0, 100, n_samples))
    }
    
    # Calculate total richness
    data['total_richness'] = data['plants_richness'] + data['animals_richness']
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some categorical features
    df['biome'] = pd.Categorical(
        np.random.choice(['forest', 'grassland', 'desert', 'tundra', 'wetland'], n_samples)
    )
    
    # Add temporal feature
    df['year'] = np.random.choice([2020, 2021, 2022, 2023], n_samples)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df.to_parquet(output_path, index=False)
    
    print(f"Generated test data with {n_samples} samples")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Saved to: {output_path}")
    
    # Show sample statistics
    print("\nTarget variable statistics:")
    print(df[['plants_richness', 'animals_richness', 'total_richness']].describe())
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate test parquet data for ML pipeline')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--output', default='outputs/test_biodiversity.parquet', help='Output path')
    parser.add_argument('--show-head', action='store_true', help='Show first few rows')
    
    args = parser.parse_args()
    
    df = generate_test_biodiversity_data(args.samples, args.output)
    
    if args.show_head:
        print("\nFirst 5 rows:")
        print(df.head())

if __name__ == '__main__':
    main()