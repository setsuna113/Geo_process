"""Core distance calculations for biodiversity analysis.

This module provides optimized distance calculations used by all analysis methods.
Single source of truth for distance mathematics - no duplication.
"""

import numpy as np
from numba import jit


# ============================================================================
# CORE DISTANCE FUNCTIONS - Single source of truth
# ============================================================================

@jit(nopython=True, cache=True, inline='always', fastmath=True)
def bray_curtis_distance_single(sample, weight, min_valid_features=2):
    """Calculate Bray-Curtis distance between two vectors.
    
    Single source of truth for Bray-Curtis distance calculation.
    Used by both training and metrics calculations.
    
    Args:
        sample: Input vector, may contain NaN
        weight: Weight vector, may contain NaN  
        min_valid_features: Minimum valid features required
        
    Returns:
        Distance value (0-1), or inf if insufficient valid features
    """
    numerator = 0.0
    denominator = 0.0
    valid_count = 0
    n_features = len(sample)
    
    for k in range(n_features):
        if not np.isnan(sample[k]) and not np.isnan(weight[k]):
            abs_diff = abs(sample[k] - weight[k])
            sum_val = sample[k] + weight[k]
            if sum_val > 0:
                numerator += abs_diff
                denominator += sum_val
                valid_count += 1
    
    if valid_count >= min_valid_features and denominator > 0:
        return numerator / denominator
    return np.inf


@jit(nopython=True, cache=True, inline='always', fastmath=True)
def euclidean_distance_single(sample, weight, min_valid_features=1):
    """Calculate Euclidean distance between two vectors.
    
    Args:
        sample: Input vector, may contain NaN
        weight: Weight vector, may contain NaN
        min_valid_features: Minimum valid features required
        
    Returns:
        Squared distance, or inf if insufficient valid features
    """
    dist = 0.0
    valid_count = 0
    n_features = len(sample)
    
    for k in range(n_features):
        if not np.isnan(sample[k]) and not np.isnan(weight[k]):
            diff = sample[k] - weight[k]
            dist += diff * diff
            valid_count += 1
    
    if valid_count >= min_valid_features:
        return dist
    return np.inf


@jit(nopython=True, cache=True, inline='always')
def haversine_distance_single(lon1, lat1, lon2, lat2):
    """Calculate haversine distance between two coordinates.
    
    Args:
        lon1, lat1: First coordinate in degrees
        lon2, lat2: Second coordinate in degrees
        
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in km (standard value, config can override if needed)
    return 6371.0 * c


# ============================================================================
# BMU FINDING - Shared logic
# ============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def find_bmu_numba(sample, weights, min_valid_features=2):
    """Find Best Matching Unit for a sample.
    
    Shared BMU finding logic used by both training and prediction.
    
    Args:
        sample: Input vector
        weights: Weight matrix (n_neurons, n_features)
        min_valid_features: Minimum valid features required
        
    Returns:
        Index of BMU, or -1 if no valid BMU found
    """
    n_neurons = weights.shape[0]
    min_dist = np.inf
    bmu_idx = -1
    
    for j in range(n_neurons):
        dist = bray_curtis_distance_single(sample, weights[j], min_valid_features)
        if dist < min_dist:
            min_dist = dist
            bmu_idx = j
    
    return bmu_idx


@jit(nopython=True, cache=True, fastmath=True)
def find_two_bmus_numba(sample, weights, min_valid_features=2):
    """Find first and second Best Matching Units.
    
    Used for topographic error calculation.
    
    Args:
        sample: Input vector
        weights: Weight matrix (n_neurons, n_features)
        min_valid_features: Minimum valid features required
        
    Returns:
        Tuple of (first_bmu_idx, second_bmu_idx)
    """
    n_neurons = weights.shape[0]
    min_dist1 = np.inf
    min_dist2 = np.inf
    bmu1_idx = -1
    bmu2_idx = -1
    
    for j in range(n_neurons):
        dist = bray_curtis_distance_single(sample, weights[j], min_valid_features)
        if dist < min_dist1:
            # Current best becomes second best
            min_dist2 = min_dist1
            bmu2_idx = bmu1_idx
            # New best
            min_dist1 = dist
            bmu1_idx = j
        elif dist < min_dist2:
            min_dist2 = dist
            bmu2_idx = j
    
    return bmu1_idx, bmu2_idx