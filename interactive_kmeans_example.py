"""Interactive example for running k-means on biodiversity data."""

import pandas as pd
import numpy as np
from src.biodiversity_analysis.methods.kmeans.analyzer import KMeansAnalyzer
from src.biodiversity_analysis.methods.kmeans.visualization import KMeansVisualizer

# Load and explore the data
data_path = "/home/yl998/dev/geo/outputs/biodiversity_data.parquet"
df = pd.read_parquet(data_path)

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Calculate missing data percentage
feature_cols = [col for col in df.columns if col not in ['latitude', 'longitude', 'x', 'y']]
missing_pct = (df[feature_cols].isna().sum() / len(df)).mean() * 100
print(f"\nAverage missing data: {missing_pct:.1f}%")

# Run k-means analysis
analyzer = KMeansAnalyzer()

# Quick analysis with default settings
result = analyzer.analyze(
    data_path=data_path,
    n_clusters=15,  # Start with 15 clusters
    save_results=True
)

# Access results
labels = result.labels
print(f"\nUnique clusters: {np.unique(labels[labels >= 0])}")

# Get cluster assignments with coordinates
coords = result.additional_outputs['coordinates']
cluster_df = pd.DataFrame({
    'latitude': [c[0] for c in coords],
    'longitude': [c[1] for c in coords],
    'cluster': labels
})

# Export results for visualization
cluster_df.to_csv("./outputs/kmeans_results/cluster_assignments.csv", index=False)
print("\nCluster assignments saved to: ./outputs/kmeans_results/cluster_assignments.csv")

# Visualize results (if matplotlib is available)
try:
    visualizer = KMeansVisualizer()
    
    # Plot geographic distribution
    fig = visualizer.plot_geographic_clusters(
        np.array(coords),
        labels
    )
    fig.savefig("./outputs/kmeans_results/geographic_clusters.png", dpi=150)
    print("Geographic visualization saved!")
    
except Exception as e:
    print(f"Visualization skipped: {e}")