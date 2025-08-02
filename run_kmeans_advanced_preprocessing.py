#!/usr/bin/env python3
"""Run k-means clustering with advanced preprocessing on biodiversity data.

This script demonstrates the implementation of cutting-edge preprocessing
techniques from the research recommendations, including:
- Spatial autocorrelation features (Moran's I, spatial lags)
- Zero-inflation handling
- Biodiversity-specific features
- Multi-scale spatial statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import List

# Import preprocessing pipeline
from src.biodiversity_analysis.preprocessing.pipeline import BiodiversityPreprocessingPipeline
from src.biodiversity_analysis.preprocessing.spatial_features import SpatialFeatureEngineering

# Import k-means implementation
from src.biodiversity_analysis.methods.kmeans import BiodiversityKMeans, KMeansAnalyzer
from src.biodiversity_analysis.shared.data.parquet_loader import ParquetDataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_biodiversity_data(data_path: str, sample_size: int = None) -> pd.DataFrame:
    """Load biodiversity data with proper coordinate handling."""
    logger.info(f"Loading data from {data_path}...")
    
    loader = ParquetDataLoader(data_path)
    df = loader.load_data()
    
    if sample_size and sample_size < len(df):
        logger.info(f"Sampling {sample_size} records...")
        df = df.sample(n=sample_size, random_state=42)
    
    return df


def extract_species_matrix(df: pd.DataFrame) -> tuple:
    """Extract species abundance matrix and metadata."""
    # Identify species columns (numeric columns excluding coordinates)
    coord_cols = ['latitude', 'longitude', 'lat', 'lon', 'x', 'y']
    species_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                    if col not in coord_cols]
    
    # Extract coordinates
    if 'latitude' in df.columns:
        coordinates = df[['latitude', 'longitude']].values
    elif 'lat' in df.columns:
        coordinates = df[['lat', 'lon']].values
    else:
        raise ValueError("No coordinate columns found")
    
    # Extract species data
    species_data = df[species_cols].values
    
    logger.info(f"Extracted {len(species_cols)} species features")
    logger.info(f"Data shape: {species_data.shape}")
    logger.info(f"Missing data: {np.isnan(species_data).sum() / species_data.size * 100:.1f}%")
    
    return species_data, coordinates, species_cols


def visualize_preprocessing_results(original_data: np.ndarray,
                                   preprocessed_data: np.ndarray,
                                   feature_names: List[str],
                                   metadata: dict,
                                   output_dir: Path):
    """Create visualizations of preprocessing effects."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # 1. Feature count comparison
    ax = axes[0]
    feature_types = ['Original', 'Zero-inflated', 'Diversity', 'Spatial', 'Total']
    counts = [
        original_data.shape[1],
        metadata['n_zero_features'],
        metadata['n_diversity_features'],
        metadata['n_spatial_features'],
        metadata['n_total_features']
    ]
    ax.bar(feature_types, counts)
    ax.set_title('Feature Count by Type')
    ax.set_ylabel('Number of Features')
    
    # 2. Missing data before/after
    ax = axes[1]
    missing_before = np.isnan(original_data).sum() / original_data.size * 100
    missing_after = np.isnan(preprocessed_data).sum() / preprocessed_data.size * 100
    ax.bar(['Before', 'After'], [missing_before, missing_after])
    ax.set_title('Missing Data %')
    ax.set_ylabel('Percentage')
    
    # 3. Spatial autocorrelation
    ax = axes[2]
    if 'global_morans_i' in metadata:
        morans_values = list(metadata['global_morans_i'].values())[:10]
        morans_names = list(metadata['global_morans_i'].keys())[:10]
        ax.barh(range(len(morans_values)), morans_values)
        ax.set_yticks(range(len(morans_values)))
        ax.set_yticklabels([n[:20] for n in morans_names], fontsize=8)
        ax.set_xlabel("Moran's I")
        ax.set_title('Spatial Autocorrelation')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # 4. Feature distributions (sample)
    ax = axes[3]
    sample_features = preprocessed_data[:, :5]
    ax.boxplot(sample_features)
    ax.set_title('Sample Feature Distributions')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Standardized Value')
    
    # 5. Correlation heatmap (sample)
    ax = axes[4]
    sample_idx = np.random.choice(preprocessed_data.shape[1], 
                                  min(20, preprocessed_data.shape[1]), 
                                  replace=False)
    corr_matrix = np.corrcoef(preprocessed_data[:, sample_idx].T)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title('Feature Correlation (Sample)')
    plt.colorbar(im, ax=ax)
    
    # 6. PCA variance explained
    ax = axes[5]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(20, preprocessed_data.shape[1]))
    pca.fit(preprocessed_data[~np.isnan(preprocessed_data).any(axis=1)])
    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('PCA Analysis')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'preprocessing_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved preprocessing analysis to {output_dir}")


def run_kmeans_with_preprocessing(data_path: str,
                                  output_dir: str,
                                  n_clusters: int = 30,
                                  sample_size: int = 100000):
    """Run k-means with advanced preprocessing pipeline."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    df = load_biodiversity_data(data_path, sample_size)
    species_data, coordinates, species_names = extract_species_matrix(df)
    
    # Save original data stats
    original_stats = {
        'n_samples': species_data.shape[0],
        'n_species': species_data.shape[1],
        'missing_proportion': float(np.isnan(species_data).sum() / species_data.size),
        'species_names': species_names
    }
    
    # Initialize preprocessing pipeline
    logger.info("Initializing preprocessing pipeline...")
    pipeline = BiodiversityPreprocessingPipeline(
        # Zero-inflation parameters
        structural_zero_threshold=0.95,
        pseudo_count_method='adaptive',
        compositional_transform='hurdle',  # NEW: Use hurdle model transformation
        
        # Spatial parameters
        spatial_neighbors=8,
        spatial_scales=[5, 10, 25, 50, 100],  # km
        n_moran_eigenvectors=20,
        use_pmem=True,  # NEW: Use predictive MEM
        pmem_dwf='exponential',  # Distance-weighting function
        
        # Diversity parameters - DISABLED for richness clustering
        calculate_interactions=False,  # Avoid circular reasoning
        calculate_diversity_indices=False,  # Don't use diversity indices as features
        environmental_features=[],  # Would add temp, precip etc if available
        
        # Standardization
        standardize=True,
        standardization_method='robust'
    )
    
    # Fit and transform data
    logger.info("Applying preprocessing pipeline...")
    preprocessed_data, feature_names, metadata = pipeline.fit_transform(
        species_data=species_data,
        coordinates=coordinates,
        species_names=species_names
    )
    
    # Visualize preprocessing results
    visualize_preprocessing_results(
        species_data, preprocessed_data, feature_names, metadata, output_dir
    )
    
    # Save preprocessing report
    report = pipeline.get_preprocessing_report()
    with open(output_dir / 'preprocessing_report.txt', 'w') as f:
        f.write(report)
    
    # Save feature metadata
    feature_metadata = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'feature_groups': metadata['feature_groups'],
        'preprocessing_params': {
            'structural_zero_threshold': 0.95,
            'spatial_scales': [5, 10, 25, 50, 100],
            'n_moran_eigenvectors': 20
        }
    }
    with open(output_dir / 'feature_metadata.json', 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    
    # Run k-means on preprocessed data
    logger.info(f"Running k-means with {n_clusters} clusters on preprocessed data...")
    
    # Configure k-means for high-dimensional data
    kmeans = BiodiversityKMeans(
        n_clusters=n_clusters,
        distance_metric='euclidean',  # Use Euclidean after preprocessing
        init_method='kmeans++',
        max_iter=300,
        tol=1e-4,
        n_init=10,
        random_state=42
    )
    
    # Handle any remaining NaN values
    valid_mask = ~np.isnan(preprocessed_data).any(axis=1)
    valid_data = preprocessed_data[valid_mask]
    valid_coords = coordinates[valid_mask]
    
    logger.info(f"Clustering {valid_data.shape[0]} samples with {valid_data.shape[1]} features")
    
    # Fit k-means
    kmeans.fit(valid_data)
    
    # Get cluster assignments
    labels = kmeans.labels_
    
    # Create full labels array with -1 for invalid samples
    full_labels = np.full(len(preprocessed_data), -1)
    full_labels[valid_mask] = labels
    
    # Save results
    results = {
        'preprocessing_metadata': metadata,
        'original_stats': original_stats,
        'n_valid_samples': int(valid_mask.sum()),
        'n_clusters': n_clusters,
        'cluster_sizes': [int((labels == i).sum()) for i in range(n_clusters)],
        'inertia': float(kmeans.inertia_),
        'n_iterations': int(kmeans.n_iter_),
        'feature_importance': {
            'n_original_features': species_data.shape[1],
            'n_preprocessed_features': preprocessed_data.shape[1],
            'feature_groups': metadata['feature_groups']
        }
    }
    
    with open(output_dir / 'clustering_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save cluster assignments with coordinates
    results_df = pd.DataFrame({
        'latitude': coordinates[:, 0],
        'longitude': coordinates[:, 1],
        'cluster': full_labels,
        'valid': valid_mask
    })
    results_df.to_csv(output_dir / 'cluster_assignments.csv', index=False)
    
    # Analyze results
    logger.info("Analyzing clustering results...")
    analyzer = KMeansAnalyzer()
    
    # Create analysis with feature groups
    feature_groups = {
        'Zero-inflation': list(range(metadata['n_zero_features'])),
        'Diversity': list(range(metadata['n_zero_features'], 
                               metadata['n_zero_features'] + metadata['n_diversity_features'])),
        'Spatial': list(range(metadata['n_zero_features'] + metadata['n_diversity_features'],
                             metadata['n_total_features']))
    }
    
    # Generate comprehensive analysis
    analyzer.analyze(
        data=valid_data,
        labels=labels,
        coordinates=valid_coords,
        feature_names=feature_names,
        output_dir=output_dir,
        feature_groups=feature_groups,
        distance_metric='euclidean'
    )
    
    logger.info(f"Analysis complete! Results saved to {output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("K-MEANS WITH ADVANCED PREPROCESSING - SUMMARY")
    print("="*60)
    print(f"Original features: {species_data.shape[1]}")
    print(f"Preprocessed features: {preprocessed_data.shape[1]}")
    print(f"  - Zero-inflation features: {metadata['n_zero_features']}")
    print(f"  - Diversity features: {metadata['n_diversity_features']}")
    print(f"  - Spatial features: {metadata['n_spatial_features']}")
    print(f"Valid samples: {valid_mask.sum()}/{len(valid_mask)}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Iterations to converge: {kmeans.n_iter_}")
    print(f"Final inertia: {kmeans.inertia_:.2f}")
    print("="*60)
    
    return kmeans, results_df, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run k-means with advanced preprocessing on biodiversity data"
    )
    parser.add_argument(
        "--data",
        default="/home/jason/geo/outputs/biodiversity_data.parquet",
        help="Path to biodiversity parquet file"
    )
    parser.add_argument(
        "--output",
        default="/home/jason/geo/outputs/kmeans_advanced",
        help="Output directory for results"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=30,
        help="Number of clusters"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100000,
        help="Sample size (default: 100000)"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    run_kmeans_with_preprocessing(
        data_path=args.data,
        output_dir=args.output,
        n_clusters=args.clusters,
        sample_size=args.sample
    )