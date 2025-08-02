#!/usr/bin/env python3
"""Debug geographic distance calculation."""

import numpy as np
import time
from src.biodiversity_analysis.methods.som.geo_som_core import GeoSOMConfig, GeoSOMVLRSOM

# Test with small data first
n_samples = 1000
n_features = 4
n_neurons = 225

print("Testing geographic distance calculation...")

# Create test data
data = np.random.rand(n_samples, n_features)
mask = np.random.rand(n_samples, n_features) < 0.948
data[mask] = np.nan

# Create coordinates
coords = np.column_stack([
    np.random.uniform(-10, 10, n_samples),
    np.random.uniform(40, 50, n_samples)
])

weights = np.random.rand(n_neurons, n_features)
neuron_coords = np.column_stack([
    np.random.uniform(-10, 10, n_neurons),
    np.random.uniform(40, 50, n_neurons)
])

# Create SOM instance
config = GeoSOMConfig(grid_size=(15, 15), spatial_weight=0.3)
som = GeoSOMVLRSOM(config)
som.weights = weights
som.neuron_coords = neuron_coords

print(f"\nTest data shape: {data.shape}")
print(f"Coordinates shape: {coords.shape}")
print(f"Weights shape: {weights.shape}")
print(f"Neuron coords shape: {neuron_coords.shape}")

# Test haversine distance calculation
print("\nTesting haversine distance calculation...")
start = time.time()
geo_distances = som._vectorized_haversine_distances(coords[:10], neuron_coords)
print(f"Haversine calculation took {time.time() - start:.4f}s")
print(f"Geographic distances shape: {geo_distances.shape}")
print(f"Distance range: {geo_distances.min():.1f} - {geo_distances.max():.1f} km")

# Test combined distance calculation
print("\nTesting combined distance calculation...")
start = time.time()
combined_distances = som._vectorized_combined_distance(
    data[:10], weights, coords[:10], neuron_coords
)
print(f"Combined distance calculation took {time.time() - start:.4f}s")
print(f"Combined distances shape: {combined_distances.shape}")
print(f"Valid distances: {(~np.isinf(combined_distances)).sum()}")

# Test full chunk processing
print("\nTesting full chunk processing (10k samples)...")
chunk_size = min(10000, n_samples)
chunk_data = data[:chunk_size]
chunk_coords = coords[:chunk_size]

start = time.time()
if hasattr(som, 'neuron_coords'):
    chunk_distances = som._vectorized_combined_distance(
        chunk_data, weights, chunk_coords, neuron_coords
    )
else:
    chunk_distances = som._vectorized_partial_bray_curtis_chunk(chunk_data, weights)
print(f"Chunk processing took {time.time() - start:.4f}s")

print("\nAll tests passed!")