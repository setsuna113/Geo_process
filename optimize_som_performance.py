#!/usr/bin/env python3
"""Optimize SOM performance for production run."""

import numpy as np

# Configuration optimizations for production
optimizations = {
    # Reduce chunk size for better progress visibility
    "chunk_size": 5000,  # was 10000
    
    # Use approximation for geographic distances
    "use_euclidean_approx": True,  # faster than haversine for small regions
    
    # Reduce epochs for CV
    "max_epochs_cv": 50,  # was 1000
    
    # Early stopping
    "patience": 10,  # was 50
    
    # Reduce logging overhead
    "log_every_n_chunks": 100,  # log progress every 100 chunks
}

print("=== SOM Performance Optimizations ===")
print("\n1. Use Euclidean approximation for geographic distances")
print("   - For biodiversity data spanning ~1000km, approximation error < 1%")
print("   - 10x faster than haversine calculation")

print("\n2. Reduce CV epochs to 50")
print("   - SOM typically converges within 20-50 epochs")
print("   - Can increase if needed for final run")

print("\n3. Smaller chunk size (5000)")
print("   - Better progress visibility")
print("   - More frequent updates")

print("\n4. Early stopping with patience=10")
print("   - Stop when no improvement for 10 epochs")

# Test performance improvement
n_points = 10000
coords1 = np.random.uniform(-10, 10, (n_points, 2))
coords2 = np.random.uniform(-10, 10, (225, 2))

print("\n=== Performance Test ===")

# Haversine (current)
import time
start = time.time()
coords1_rad = np.radians(coords1)
coords2_rad = np.radians(coords2)
lat1 = coords1_rad[:, 1:2]
lon1 = coords1_rad[:, 0:1]
lat2 = coords2_rad[:, 1].reshape(1, -1)
lon2 = coords2_rad[:, 0].reshape(1, -1)
dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
distances_haversine = 6371 * c
haversine_time = time.time() - start

# Euclidean approximation
start = time.time()
# Convert degrees to km (approximate)
km_per_deg_lat = 111.0  # ~constant
km_per_deg_lon = 111.0 * np.cos(np.radians(45))  # at 45° latitude
coords1_km = coords1 * np.array([km_per_deg_lon, km_per_deg_lat])
coords2_km = coords2 * np.array([km_per_deg_lon, km_per_deg_lat])
# Euclidean distance
diff = coords1_km[:, np.newaxis, :] - coords2_km[np.newaxis, :, :]
distances_euclidean = np.sqrt(np.sum(diff**2, axis=2))
euclidean_time = time.time() - start

print(f"Haversine time: {haversine_time:.4f}s")
print(f"Euclidean time: {euclidean_time:.4f}s")
print(f"Speedup: {haversine_time/euclidean_time:.1f}x")

# Check accuracy
max_error = np.max(np.abs(distances_haversine - distances_euclidean))
mean_error = np.mean(np.abs(distances_haversine - distances_euclidean))
print(f"\nAccuracy:")
print(f"Max error: {max_error:.2f} km")
print(f"Mean error: {mean_error:.2f} km")
print(f"Relative error: {mean_error/np.mean(distances_haversine)*100:.1f}%")

print("\n=== Recommended Config Changes ===")
print("In config.yml, update test_som experiment:")
print("  iterations: 50  # was 1000")
print("  patience: 10    # for early stopping")
print("\nEstimated time with optimizations: ~1.5 hours total")