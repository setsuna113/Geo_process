#!/usr/bin/env python3
"""
Generate test biodiversity data for SOM analysis testing.
Creates a small parquet file with realistic biodiversity patterns and some NaN values.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_biodiversity_data(n_samples=1000, output_path='outputs/test_biodiversity.parquet'):
    """Generate synthetic biodiversity data with realistic patterns."""
    
    logger.info(f"Generating {n_samples} samples of test biodiversity data")
    
    # Create realistic geographic coordinates (focused on a region)
    np.random.seed(42)  # For reproducibility
    
    # Simulate a region roughly covering part of Europe/North America
    lon_min, lon_max = -10, 30  # Longitude range
    lat_min, lat_max = 35, 65   # Latitude range
    
    longitude = np.random.uniform(lon_min, lon_max, n_samples)
    latitude = np.random.uniform(lat_min, lat_max, n_samples)
    
    # Create biodiversity features with spatial correlation
    # Use distance from center to create gradients
    center_lon, center_lat = 10, 50
    distance_from_center = np.sqrt((longitude - center_lon)**2 + (latitude - center_lat)**2)
    
    # Temperature gradient (decreases with latitude and distance from center)
    avg_temp = 20 - 0.5 * (latitude - lat_min) - 0.1 * distance_from_center + np.random.normal(0, 2, n_samples)
    seasonal_temp = 15 + 0.3 * (latitude - lat_min) + np.random.normal(0, 1.5, n_samples)
    
    # Precipitation (varies with longitude and some randomness)
    avg_precip = 800 + 10 * (longitude - lon_min) - 0.05 * distance_from_center**2 + np.random.normal(0, 100, n_samples)
    avg_precip = np.maximum(avg_precip, 100)  # Ensure minimum precipitation
    
    # Species richness (correlated with favorable climate conditions)
    species_richness = (50 + 
                       2 * np.maximum(avg_temp - 5, 0) +  # Warmer is better (up to a point)
                       0.01 * avg_precip +                # More rain is better
                       -0.1 * distance_from_center +      # Center is more diverse
                       np.random.normal(0, 8, n_samples))
    species_richness = np.maximum(species_richness, 5).astype(int)  # Minimum 5 species
    
    # Shannon diversity (related to richness but with some independence)
    shannon_diversity = (1.5 + 
                        0.05 * species_richness + 
                        0.02 * avg_precip/100 +
                        np.random.normal(0, 0.3, n_samples))
    shannon_diversity = np.maximum(shannon_diversity, 0.5)
    
    # Endemism count (more endemic species in isolated/diverse areas)
    endemic_count = (np.random.poisson(5, n_samples) + 
                    0.1 * species_richness + 
                    0.001 * distance_from_center**2).astype(int)
    
    # Functional diversity (correlated with richness and environment)
    functional_diversity = (0.6 + 
                           0.01 * species_richness +
                           0.001 * avg_precip +
                           np.random.normal(0, 0.15, n_samples))
    functional_diversity = np.maximum(functional_diversity, 0.1)
    
    # Phylogenetic diversity 
    phylogenetic_diversity = (15 + 
                             0.3 * species_richness +
                             0.5 * functional_diversity +
                             np.random.normal(0, 3, n_samples))
    phylogenetic_diversity = np.maximum(phylogenetic_diversity, 5)
    
    # Create DataFrame
    data = {
        'longitude': longitude,
        'latitude': latitude,
        'avg_temp': avg_temp,
        'seasonal_temp': seasonal_temp, 
        'avg_precip': avg_precip,
        'species_richness': species_richness,
        'shannon_diversity': shannon_diversity,
        'endemic_count': endemic_count,
        'functional_diversity': functional_diversity,
        'phylogenetic_diversity': phylogenetic_diversity
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some realistic NaN patterns to test our imputation fixes
    logger.info("Introducing NaN values to test imputation strategies")
    
    # 5% missing values in temperature (sensor failures)
    temp_missing = np.random.choice(n_samples, int(0.05 * n_samples), replace=False)
    df.loc[temp_missing, 'avg_temp'] = np.nan
    
    # 3% missing values in precipitation (weather station gaps)
    precip_missing = np.random.choice(n_samples, int(0.03 * n_samples), replace=False)
    df.loc[precip_missing, 'avg_precip'] = np.nan
    
    # 8% missing values in diversity indices (harder to measure)
    div_missing = np.random.choice(n_samples, int(0.08 * n_samples), replace=False)
    df.loc[div_missing, 'shannon_diversity'] = np.nan
    
    # 2% missing values in functional diversity (very specialized measurement)
    func_missing = np.random.choice(n_samples, int(0.02 * n_samples), replace=False)
    df.loc[func_missing, 'functional_diversity'] = np.nan
    
    # Add one completely empty column to test our column removal logic
    df['empty_column'] = np.nan
    
    # Add some binary columns to test mode imputation
    # Protected area status (binary)
    protected_area = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    # Add some missing values
    protected_missing = np.random.choice(n_samples, int(0.04 * n_samples), replace=False)
    protected_area = protected_area.astype(float)  # Convert to float to allow NaN
    protected_area[protected_missing] = np.nan
    df['protected_area'] = protected_area
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    logger.info(f"Saving test data to: {output_path}")
    df.to_parquet(output_path, index=False)
    
    # Log summary statistics
    logger.info("Test data summary:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Missing values per column:")
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            logger.info(f"    {col}: {missing_count} ({missing_count/len(df)*100:.1f}%)")
    
    logger.info(f"  Feature column ranges:")
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ['longitude', 'latitude']:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                logger.info(f"    {col}: {valid_data.min():.2f} to {valid_data.max():.2f}")
    
    return output_path

if __name__ == '__main__':
    output_file = generate_test_biodiversity_data()
    print(f"✅ Test data generated: {output_file}")