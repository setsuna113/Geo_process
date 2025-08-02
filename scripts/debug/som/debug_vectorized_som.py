#!/usr/bin/env python3
"""Debug why vectorized SOM is slow."""

import numpy as np
import time

# Test the vectorized operations
n_samples = 9_000_000
n_neurons = 225
n_features = 4

print("Testing vectorized operations timing...")

# Create test data with 94.8% missing
data = np.random.rand(n_samples, n_features)
mask = np.random.rand(n_samples, n_features) < 0.948
data[mask] = np.nan

weights = np.random.rand(n_neurons, n_features)

print(f"\nData shape: {data.shape}")
print(f"Weights shape: {weights.shape}")
print(f"Missing data: {np.isnan(data).mean():.1%}")

# Test expanding dimensions
print("\nStep 1: Expanding dimensions...")
start = time.time()
data_exp = data[:, np.newaxis, :]  # This creates a (9M, 1, 4) array
print(f"Time: {time.time() - start:.2f}s")
print(f"Expanded shape: {data_exp.shape}")
print(f"Memory size: {data_exp.nbytes / 1e9:.1f} GB")

# This is the problem - we're creating massive intermediate arrays!
# With 9M samples, this expansion alone needs huge memory