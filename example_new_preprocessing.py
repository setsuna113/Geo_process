#!/usr/bin/env python3
"""Example demonstrating new preprocessing features for both k-means and SOM.

This shows how to use:
1. Hurdle model transformation for zero-inflated data
2. Predictive Moran's Eigenvector Maps (pMEM)
3. Disabled diversity indices to avoid circular reasoning
"""

import numpy as np
from src.biodiversity_analysis.preprocessing.pipeline import BiodiversityPreprocessingPipeline

# Example usage
def preprocess_for_clustering(species_data, coordinates):
    """Preprocess biodiversity data for k-means or SOM clustering."""
    
    # Initialize pipeline with new features
    pipeline = BiodiversityPreprocessingPipeline(
        # NEW: Hurdle model for zero-inflation
        compositional_transform='hurdle',  # Separates presence/absence from abundance
        
        # NEW: Predictive MEM for better spatial patterns
        use_pmem=True,
        pmem_dwf='exponential',  # or 'gaussian', 'linear'
        
        # IMPORTANT: Disable diversity indices for richness clustering
        calculate_diversity_indices=False,  # Avoid circular reasoning
        
        # Standard preprocessing options
        structural_zero_threshold=0.95,
        pseudo_count_method='adaptive',
        spatial_neighbors=8,
        spatial_scales=[5, 10, 25, 50, 100],
        n_moran_eigenvectors=20,
        standardize=True,
        standardization_method='robust'
    )
    
    # Transform data
    preprocessed_data, feature_names, metadata = pipeline.fit_transform(
        species_data=species_data,
        coordinates=coordinates
    )
    
    print(f"Original features: {species_data.shape[1]}")
    print(f"Preprocessed features: {preprocessed_data.shape[1]}")
    print(f"  - Zero-inflation features: {metadata['n_zero_features']}")
    print(f"  - Spatial features: {metadata['n_spatial_features']}")
    
    return preprocessed_data, feature_names, metadata


# The preprocessed data can be used with either k-means or SOM:
def use_with_kmeans(preprocessed_data):
    """Example: Use preprocessed data with k-means."""
    from src.biodiversity_analysis.methods.kmeans import BiodiversityKMeans
    
    kmeans = BiodiversityKMeans(
        n_clusters=30,
        distance_metric='euclidean',  # Use Euclidean after preprocessing
        init_method='kmeans++',
        max_iter=300
    )
    
    # Remove any NaN values
    valid_mask = ~np.isnan(preprocessed_data).any(axis=1)
    valid_data = preprocessed_data[valid_mask]
    
    kmeans.fit(valid_data)
    return kmeans.labels_


def use_with_som(preprocessed_data):
    """Example: Use preprocessed data with SOM."""
    from src.spatial_analysis.som import GeoSOMAnalyzer
    
    som = GeoSOMAnalyzer(
        grid_size=(10, 10),
        learning_rate=0.5,
        neighborhood_function='gaussian',
        iterations=1000
    )
    
    # Remove any NaN values
    valid_mask = ~np.isnan(preprocessed_data).any(axis=1)
    valid_data = preprocessed_data[valid_mask]
    
    som.fit(valid_data)
    return som.get_bmus()  # Best Matching Units


if __name__ == "__main__":
    # Example with synthetic data
    n_samples = 1000
    n_species = 50
    
    # Create synthetic zero-inflated data (95% zeros)
    species_data = np.random.poisson(0.1, (n_samples, n_species))
    species_data[species_data > 0] = np.random.poisson(5, np.sum(species_data > 0))
    
    # Create random coordinates
    coordinates = np.random.randn(n_samples, 2) * 10
    
    # Preprocess data
    preprocessed_data, feature_names, metadata = preprocess_for_clustering(
        species_data, coordinates
    )
    
    print("\nPreprocessing complete!")
    print(f"Hurdle transformation created {metadata['n_zero_features']} features")
    print(f"pMEM created spatial patterns optimized for prediction")
    print("\nData is now ready for k-means or SOM clustering!")