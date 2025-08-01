#!/usr/bin/env python3
"""Run k-means clustering analysis on biodiversity data."""

import sys
import os
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.biodiversity_analysis.methods.kmeans.analyzer import KMeansAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Data path
    data_path = "/home/yl998/dev/geo/outputs/biodiversity_data.parquet"
    
    # Initialize analyzer
    analyzer = KMeansAnalyzer()
    
    # Run analysis with custom parameters
    result = analyzer.analyze(
        data_path=data_path,
        n_clusters=20,  # Number of clusters
        determine_k=True,  # Automatically determine optimal k
        k_range=range(10, 31),  # Range of k values to test
        save_results=True,  # Save results to disk
        output_dir="./outputs/kmeans_results",  # Output directory
        distance_metric='bray_curtis',  # Use Bray-Curtis for biodiversity
        weight_method='auto',  # Automatic feature weighting
        prefilter_empty=True,  # Remove empty grids
        min_features_prefilter=1,  # Minimum features required
    )
    
    # Print summary
    print(f"\nAnalysis completed!")
    print(f"Number of clusters: {result.statistics['n_clusters']}")
    print(f"Silhouette score: {result.statistics.get('silhouette_score', 'N/A')}")
    print(f"\nCluster sizes:")
    for k, size in result.statistics['cluster_sizes'].items():
        print(f"  Cluster {k}: {size} samples")
    
    print(f"\nResults saved to: ./outputs/kmeans_results/")

if __name__ == "__main__":
    main()