#!/usr/bin/env python
"""
Test script for distribution transformation analysis.

LINUS: "Test or die."
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow as pa

# Import the method directly for testing
from src.biodiversity_analysis.statistical.distribution_transformation import analyze_distribution_transformation

# Create synthetic test data with different distributions
np.random.seed(42)

# Generate test data
n_samples = 1000
data = {
    'normal_feature': np.random.normal(100, 15, n_samples),
    'skewed_positive': np.random.lognormal(3, 1.5, n_samples),
    'skewed_negative': -np.random.lognormal(3, 1.5, n_samples),
    'exponential_feature': np.random.exponential(5, n_samples),
    'uniform_feature': np.random.uniform(0, 100, n_samples),
    'bimodal_feature': np.concatenate([
        np.random.normal(30, 5, n_samples//2),
        np.random.normal(70, 5, n_samples//2)
    ])
}

# Create DataFrame
df = pd.DataFrame(data)

# Add some coordinates (will be filtered out)
df['x'] = np.random.uniform(-180, 180, n_samples)
df['y'] = np.random.uniform(-90, 90, n_samples)

# Convert to Arrow Table (mimics the actual data format)
table = pa.Table.from_pandas(df)

print(f"Created test data with {len(df)} samples and {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")
print("\nData summary:")
print(df.describe())

# Test the method
output_dir = Path("test_output/distribution_transformation")
output_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print("TESTING DISTRIBUTION TRANSFORMATION ANALYSIS")
print("="*60)

# Test with default settings
result = analyze_distribution_transformation(
    data=table,
    output_dir=output_dir,
    transforms=['raw', 'sqrt', 'log', 'boxcox']
)

print(f"\nAnalysis complete!")
print(f"Success: {result['success']}")
print(f"Features analyzed: {result['features_analyzed']}")
print(f"Transforms applied: {result['transforms_applied']}")
print(f"Results saved to: {result['results_path']}")

print("\n" + "="*60)
print("SUMMARY OF BEST TRANSFORMATIONS")
print("="*60)

for feature, stats in result['summary'].items():
    print(f"\n{feature}:")
    print(f"  Original skewness: {stats['original_skew']:.3f}")
    print(f"  Best transform: {stats['best_transform']}")
    print(f"  Best p-value: {stats['best_p']:.4f}")
    print(f"  Normal? {'YES' if stats['best_p'] > 0.05 else 'NO'}")

print("\n" + "="*60)
print("Check the output directory for plots:")
print(f"  {output_dir}")
print("="*60)