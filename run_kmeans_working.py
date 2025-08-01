#!/usr/bin/env python3
"""Working k-means analysis script with proper error handling."""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.biodiversity_analysis.methods.kmeans.analyzer import KMeansAnalyzer

def main():
    print("=== K-means Analysis for Biodiversity Data ===\n")
    
    # Data path
    data_path = "/home/jason/geo/data/biodiversity_data.parquet"
    
    # Load and prepare data
    print("1. Loading data...")
    df = pd.read_parquet(data_path, engine='fastparquet')
    print(f"   Total samples: {len(df):,}")
    
    # Sample for initial test
    sample_size = 100000
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Rename columns for compatibility
    df_sample = df_sample.rename(columns={'x': 'longitude', 'y': 'latitude'})
    
    # Save sampled data
    sample_path = "/home/jason/geo/data/biodiversity_sample_100k.parquet"
    df_sample.to_parquet(sample_path, engine='fastparquet')
    
    print(f"   Using {sample_size:,} samples for initial test")
    print(f"   Missing data: {(df_sample.isna().sum() / len(df_sample) * 100).mean():.1f}% average")
    
    # Initialize analyzer
    print("\n2. Initializing analyzer...")
    analyzer = KMeansAnalyzer()
    
    # Run analysis with reasonable parameters
    print("\n3. Running k-means clustering...")
    print("   Parameters:")
    print("   - Clusters: 10")
    print("   - Distance: Bray-Curtis (for biodiversity data)")
    print("   - Max iterations: 50")
    print("   - Initializations: 3")
    
    result = analyzer.analyze(
        data_path=sample_path,
        # Algorithm parameters
        n_clusters=10,
        determine_k=False,  # Skip for initial test
        save_results=True,
        output_dir="./outputs/kmeans_biodiversity",
        # Distance and preprocessing
        distance_metric='bray_curtis',
        weight_method='completeness',
        prefilter_empty=True,
        min_features_prefilter=1,
        # Moderate settings for balance
        max_iter=50,
        n_init=3,
        tol=0.001,
        # Performance
        n_jobs=-1,
        calculate_silhouette=False,  # Skip to avoid NaN issues
        # Handle missing data
        adaptive_mode='latitude',
        arctic_min_features=1,
        temperate_min_features=1,
        tropical_min_features=1,
    )
    
    # Display results
    print("\n4. Results:")
    print(f"   ✓ Analysis completed successfully!")
    print(f"   Number of clusters: {result.statistics['n_clusters']}")
    
    print(f"\n   Cluster sizes:")
    cluster_sizes = result.statistics['cluster_sizes']
    total_assigned = sum(cluster_sizes.values())
    for k in sorted(cluster_sizes.keys()):
        size = cluster_sizes[k]
        pct = (size / total_assigned) * 100
        print(f"     Cluster {k}: {size:,} samples ({pct:.1f}%)")
    
    # Missing data analysis
    print(f"\n   Missing data by cluster:")
    missing_by_cluster = result.statistics.get('missing_data_by_cluster', {})
    for k in sorted(missing_by_cluster.keys()):
        print(f"     Cluster {k}: {missing_by_cluster[k]*100:.1f}% missing")
    
    print(f"\n5. Output saved to: ./outputs/kmeans_biodiversity/")
    print("   - kmeans_results.json: Complete results")
    print("   - cluster_assignments.csv: Sample locations with clusters")
    
    # Suggestions for next steps
    print("\n6. Next steps:")
    print("   a) Run with determine_k=True to find optimal clusters")
    print("   b) Increase sample size or run on full dataset")
    print("   c) Visualize results using the output files")
    
    # Clean up
    os.remove(sample_path)

if __name__ == "__main__":
    try:
        # Suppress warnings for cleaner output
        import warnings
        warnings.filterwarnings('ignore')
        
        main()
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()