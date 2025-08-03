#!/usr/bin/env python
"""Test advanced preprocessing with SOM analyzer."""

import numpy as np
import os
import sys

# Set the SOM config path
os.environ['SOM_CONFIG_PATH'] = '/home/yl998/dev/geo/src/config/som/som_config_optimized.yml'

from src.biodiversity_analysis.methods.som.analyzer import GeoSOMAnalyzer
from src.biodiversity_analysis.preprocessing.pipeline import BiodiversityPreprocessingPipeline

def test_preprocessing_directly():
    """Test the preprocessing pipeline directly."""
    print("\n=== Testing Preprocessing Pipeline Directly ===")
    
    # Create synthetic data similar to your richness data
    np.random.seed(42)
    n_samples = 1000
    n_species = 4
    
    # Create 95% zero-inflated data
    species_data = np.random.poisson(0.5, (n_samples, n_species))
    zero_mask = np.random.random((n_samples, n_species)) < 0.95
    species_data[zero_mask] = 0
    
    # Create coordinates
    coordinates = np.random.randn(n_samples, 2) * 10
    
    # Species names
    species_names = ['plants', 'terrestrial', 'am_fungi', 'ecm_fungi']
    
    print(f"Input data shape: {species_data.shape}")
    print(f"Proportion of zeros: {(species_data == 0).mean():.2%}")
    print(f"Max values: {species_data.max(axis=0)}")
    
    # Test preprocessing
    preprocessor = BiodiversityPreprocessingPipeline(
        compositional_transform='hurdle',
        spatial_neighbors=8,
        spatial_scales=[10, 50, 100],
        n_moran_eigenvectors=10,
        use_pmem=True,
        calculate_diversity_indices=False,
        standardize=True,
        standardization_method='robust'
    )
    
    # Fit and transform
    transformed, feature_names, metadata = preprocessor.fit_transform(
        species_data=species_data,
        coordinates=coordinates,
        species_names=species_names
    )
    
    print(f"\nOutput shape: {transformed.shape}")
    print(f"Number of features created: {len(feature_names)}")
    print(f"Metadata: {metadata}")
    
    # Show some feature names
    print(f"\nFirst 10 feature names:")
    for i, name in enumerate(feature_names[:10]):
        print(f"  {i}: {name}")
    
    print(f"\nLast 10 feature names:")
    for i, name in enumerate(feature_names[-10:], len(feature_names)-10):
        print(f"  {i}: {name}")
    
    return transformed, feature_names


def test_som_with_preprocessing():
    """Test SOM analyzer with advanced preprocessing."""
    print("\n=== Testing SOM Analyzer Configuration ===")
    
    # Create analyzer
    analyzer = GeoSOMAnalyzer()
    
    # Check configuration
    print(f"Use advanced preprocessing: {analyzer.use_advanced_preprocessing}")
    print(f"Preprocessing config: {analyzer.config.get('preprocessing_config', {})}")
    
    if analyzer.use_advanced_preprocessing:
        print("\n✅ Advanced preprocessing is ENABLED")
        print(f"Compositional transform: {analyzer.preprocessor.zero_preprocessor.compositional_transform}")
        print(f"Spatial scales: {analyzer.preprocessor.spatial_engineer.scales}")
        print(f"Number of neighbors: {analyzer.preprocessor.spatial_engineer.n_neighbors}")
    else:
        print("\n❌ Advanced preprocessing is DISABLED")


def main():
    """Run all tests."""
    print("Testing Advanced Preprocessing Integration")
    print("=" * 50)
    
    # Test preprocessing directly
    transformed_data, feature_names = test_preprocessing_directly()
    
    # Test SOM configuration
    test_som_with_preprocessing()
    
    print("\n" + "=" * 50)
    print("✅ Testing complete!")
    print(f"Advanced preprocessing creates {transformed_data.shape[1]} features from 4 original features")
    print("This includes:")
    print("  - Hurdle transform features (presence/absence + abundance)")
    print("  - Spatial lag features")
    print("  - Multi-scale spatial statistics")
    print("  - Predictive Moran eigenvectors")


if __name__ == "__main__":
    main()