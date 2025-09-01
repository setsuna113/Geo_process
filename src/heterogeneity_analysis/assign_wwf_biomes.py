#!/usr/bin/env python
"""
Assign WWF Terrestrial Ecoregion biomes to biodiversity data points
This is a workaround for R's sf package installation issues
"""

import pandas as pd
import geopandas as gpd
import pyarrow.feather as feather
from shapely.geometry import Point
import numpy as np
import sys
import os

def assign_wwf_biomes(data_path, wwf_shapefile, output_path, sample_size=500000):
    """
    Assign WWF biomes to biodiversity data points
    
    Parameters:
    -----------
    data_path : str
        Path to biodiversity data (arrow/feather format)
    wwf_shapefile : str
        Path to WWF ecoregions shapefile
    output_path : str
        Path to save data with biome assignments
    sample_size : int
        Number of samples to use for analysis
    """
    
    print(f"Loading biodiversity data from: {data_path}")
    # Read arrow file
    df = feather.read_feather(data_path)
    print(f"Total samples: {len(df):,}")
    
    # Standardize column names
    print("Standardizing column names...")
    
    # Handle fungi columns
    if 'fungi_richness' not in df.columns:
        if 'am_fungi_richness' in df.columns and 'ecm_fungi_richness' in df.columns:
            df['fungi_richness'] = df['am_fungi_richness'] + df['ecm_fungi_richness']
            print("  Combined AM and ECM fungi into total fungi richness")
    
    # Handle vertebrates column
    if 'vertebrates_richness' not in df.columns:
        if 'terrestrial_richness' in df.columns:
            df['vertebrates_richness'] = df['terrestrial_richness']
            print("  Using terrestrial_richness as vertebrates_richness")
    
    # Handle coordinates
    if 'longitude' not in df.columns:
        if 'x' in df.columns:
            df['longitude'] = df['x']
            df['latitude'] = df['y']
            print("  Using x/y as longitude/latitude")
    
    # Sample if too large
    if len(df) > sample_size:
        print(f"\nDataset is very large ({len(df):,} rows)")
        print(f"Sampling {sample_size:,} rows for analysis...")
        np.random.seed(42)
        df = df.sample(n=sample_size)
    
    print(f"\nProceeding with {len(df):,} samples")
    
    # Remove rows with missing biodiversity data
    required_cols = ['fungi_richness', 'plants_richness', 'vertebrates_richness']
    n_before = len(df)
    df = df.dropna(subset=required_cols)
    n_after = len(df)
    if n_before > n_after:
        print(f"Removed {n_before - n_after:,} rows with missing biodiversity data")
        print(f"Continuing with {n_after:,} complete samples")
    
    # Load WWF shapefile
    print(f"\nLoading WWF ecoregions from: {wwf_shapefile}")
    wwf = gpd.read_file(wwf_shapefile)
    print(f"Loaded {len(wwf):,} ecoregion polygons")
    
    # Create geometry column for biodiversity data
    print("\nCreating spatial points from coordinates...")
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Spatial join with WWF ecoregions
    print("Assigning sites to WWF biomes (this may take a few minutes)...")
    print("Using spatial index for faster processing...")
    
    # Ensure spatial index exists
    wwf_subset = wwf[['BIOME_NUM', 'BIOME_NAME', 'geometry']].copy()
    wwf_subset.sindex  # Build spatial index if not exists
    
    gdf_with_biomes = gpd.sjoin(gdf, wwf_subset, 
                                 how='left', predicate='intersects')
    
    # Add biome information back to dataframe
    df['biome_code'] = gdf_with_biomes['BIOME_NUM']
    df['biome_name'] = gdf_with_biomes['BIOME_NAME']
    
    # Check for missing assignments
    n_missing = df['biome_code'].isna().sum()
    if n_missing > 0:
        print(f"\nWarning: {n_missing:,} sites ({n_missing/len(df)*100:.1f}%) could not be assigned to terrestrial biomes")
        print("These are likely marine or lake sites")
        print("Removing unassigned sites...")
        df = df[df['biome_code'].notna()]
        print(f"Continuing with {len(df):,} terrestrial sites")
    
    # Print biome distribution
    print("\nBiome distribution:")
    biome_counts = df['biome_name'].value_counts()
    for biome, count in biome_counts.items():
        print(f"  {biome}: {count:,} sites")
    
    print(f"\nAssigned {len(df):,} sites to {df['biome_code'].nunique()} biomes")
    
    # Save the data with biome assignments
    print(f"\nSaving data with biome assignments to: {output_path}")
    feather.write_feather(df, output_path)
    print("Done!")
    
    return df

if __name__ == "__main__":
    # Default paths
    data_path = "/scratch/yl998/biodiversity_climate_production_1x_corrected.arrow"
    wwf_shapefile = "/scratch/yl998/data/wwf_ecoregions/Ecoregions2017.shp"
    output_path = "/scratch/yl998/biodiversity_with_biomes.arrow"
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    if len(sys.argv) > 2:
        wwf_shapefile = sys.argv[2]
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    
    # Run the assignment
    assign_wwf_biomes(data_path, wwf_shapefile, output_path)