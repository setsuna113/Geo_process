#!/usr/bin/env python3
"""
Simple, direct biodiversity analysis runner.

No factories. No abstractions. Just import and run.

Usage:
    python run_biodiversity_analysis.py data.parquet --method cpwopt
    python run_biodiversity_analysis.py data.parquet --method som
    python run_biodiversity_analysis.py data.parquet --method kmeans_hierarchical
"""

import pandas as pd
import argparse
from pathlib import Path


def run_cpwopt(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run CP-WOPT tensor decomposition."""
    from src.biodiversity_analysis.methods.cpwopt import create_cpwopt_analyzer
    
    analyzer = create_cpwopt_analyzer('cpwopt_analysis')
    result = analyzer.analyze(df)
    
    if result.success:
        print(f"✓ CP-WOPT completed. Metrics: {result.data.get('metrics', {})}")
    else:
        print(f"✗ CP-WOPT failed: {result.error}")
    
    return result.data if result.success else {}


def run_som(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run Self-Organizing Map analysis."""
    from src.biodiversity_analysis.methods.som import SOMAnalyzer
    from src.config.som import SOMConfig
    
    config = SOMConfig().load_config()
    analyzer = SOMAnalyzer(config)
    result = analyzer.analyze(df)
    
    if result.success:
        print(f"✓ SOM completed. Map size: {result.data.get('map_size', 'unknown')}")
    else:
        print(f"✗ SOM failed: {result.error}")
    
    return result.data if result.success else {}


def run_kmeans_hierarchical(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run hierarchical K-means clustering."""
    from src.biodiversity_analysis.methods.kmeans_hierarchical import KMeansHierarchicalAnalyzerAdapter
    from src.config.kmeans_hierarchical import KMeansHierarchicalConfigLoader
    
    config = KMeansHierarchicalConfigLoader().load_config()
    analyzer = KMeansHierarchicalAnalyzerAdapter(config)
    result = analyzer.analyze(df)
    
    if result.success:
        print(f"✓ K-means completed. Clusters: {result.data.get('n_clusters', 'unknown')}")
    else:
        print(f"✗ K-means failed: {result.error}")
    
    return result.data if result.success else {}


def main():
    parser = argparse.ArgumentParser(description='Run biodiversity analysis')
    parser.add_argument('input', type=Path, help='Input parquet file')
    parser.add_argument('--method', choices=['cpwopt', 'som', 'kmeans_hierarchical'], 
                       default='cpwopt', help='Analysis method')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'),
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"Error: Input file {args.input} not found")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} records with {len(df.columns)} columns")
    
    # Run analysis
    print(f"\nRunning {args.method} analysis...")
    
    if args.method == 'cpwopt':
        result = run_cpwopt(df, args.output_dir)
    elif args.method == 'som':
        result = run_som(df, args.output_dir)
    elif args.method == 'kmeans_hierarchical':
        result = run_kmeans_hierarchical(df, args.output_dir)
    
    print("\nAnalysis complete!")
    return 0


if __name__ == '__main__':
    exit(main())