#!/usr/bin/env python3
"""Check memory usage of vectorized implementation."""

import numpy as np

n_samples = 9_000_000  # CV training size
n_neurons = 225  # 15x15 grid
n_features = 4

print("Memory usage estimates for vectorized implementation:")
print(f"Samples: {n_samples:,}")
print(f"Neurons: {n_neurons}")
print(f"Features: {n_features}")

# Distance matrix
distance_matrix_size = n_samples * n_neurons * 8  # float64
print(f"\nDistance matrix: {distance_matrix_size / 1e9:.1f} GB")

# Expanded data arrays
data_exp_size = n_samples * 1 * n_features * 8
weights_exp_size = 1 * n_neurons * n_features * 8
print(f"Data expanded: {data_exp_size / 1e9:.1f} GB")
print(f"Weights expanded: {weights_exp_size / 1e9:.1f} GB")

# Valid mask
valid_mask_size = n_samples * n_neurons * n_features * 1  # bool
print(f"Valid mask: {valid_mask_size / 1e9:.1f} GB")

# Influences matrix
influences_size = n_samples * n_neurons * 8
print(f"Influences matrix: {influences_size / 1e9:.1f} GB")

# Total
total = distance_matrix_size + data_exp_size + weights_exp_size + valid_mask_size + influences_size
print(f"\nTotal memory needed: {total / 1e9:.1f} GB")

print("\nThis explains the 86GB memory usage!")
print("We need a chunked version for large datasets.")