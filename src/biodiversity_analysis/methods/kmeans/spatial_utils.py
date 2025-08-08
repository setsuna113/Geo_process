"""Spatial utilities for kmeans module."""

import numpy as np
from scipy import spatial
from typing import Tuple

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points."""
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def construct_spatial_weights_matrix(coords: np.ndarray, k: int = 8) -> np.ndarray:
    """Construct spatial weights matrix using k-nearest neighbors."""
    n = len(coords)
    tree = spatial.KDTree(coords)
    
    # Find k nearest neighbors for each point
    weights = np.zeros((n, n))
    for i in range(n):
        distances, indices = tree.query(coords[i], k=k+1)  # +1 to exclude self
        for j, idx in enumerate(indices[1:]):  # Skip self
            weights[i, idx] = 1.0 / (1 + distances[j+1])  # Distance decay
    
    # Make symmetric
    weights = (weights + weights.T) / 2
    
    # Row normalize
    row_sums = weights.sum(axis=1)
    weights[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
    
    return weights

def calculate_morans_i(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    """Calculate Moran's I spatial autocorrelation statistic."""
    n = len(values)
    
    # Deviation from mean
    mean_val = values.mean()
    deviations = values - mean_val
    
    # Calculate Moran's I
    numerator = np.sum(weights * np.outer(deviations, deviations))
    denominator = np.sum(deviations ** 2)
    
    W = np.sum(weights)
    morans_i = (n / W) * (numerator / denominator) if denominator > 0 else 0
    
    # Expected value under null hypothesis
    expected_i = -1 / (n - 1)
    
    return morans_i, expected_i
