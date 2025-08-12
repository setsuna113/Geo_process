"""SOM-specific quality metrics.

This module provides optimized calculations for SOM quality metrics like
Quantization Error (QE) and Topographic Error (TE). These are NOT distances
but quality measures that USE distances.
"""

import numpy as np
from numba import jit, prange
from .distance_calculator import (
    bray_curtis_distance_single,
    find_bmu_numba,
    find_two_bmus_numba,
    haversine_distance_single
)


# ============================================================================
# QUANTIZATION ERROR - Direct calculation without matrices
# ============================================================================

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def calculate_qe_numba(data, weights, min_valid_features=2):
    """Calculate Quantization Error directly in parallel.
    
    No intermediate distance matrices - memory efficient.
    
    Args:
        data: Input data (n_samples, n_features)
        weights: SOM weights (n_neurons, n_features)
        min_valid_features: Minimum valid features required
        
    Returns:
        Average quantization error
    """
    n_samples = data.shape[0]
    total_error = 0.0
    valid_count = 0
    
    # Parallel processing of samples
    for i in prange(n_samples):
        sample = data[i]
        
        # Find BMU and its distance
        min_dist = np.inf
        for j in range(weights.shape[0]):
            dist = bray_curtis_distance_single(sample, weights[j], min_valid_features)
            if dist < min_dist:
                min_dist = dist
        
        # Accumulate valid errors
        if not np.isinf(min_dist):
            total_error += min_dist
            valid_count += 1
    
    return total_error / valid_count if valid_count > 0 else np.inf


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def calculate_qe_with_coords_numba(data, weights, coordinates, neuron_coords, 
                                   spatial_weight=0.2, min_valid_features=2,
                                   geo_norm=20000.0):
    """Calculate QE with geographic distance component.
    
    Args:
        data: Input data (n_samples, n_features)
        weights: SOM weights (n_neurons, n_features)
        coordinates: Sample coordinates (n_samples, 2) [lon, lat]
        neuron_coords: Neuron coordinates (n_neurons, 2) [lon, lat]
        spatial_weight: Weight for geographic distance (0-1)
        min_valid_features: Minimum valid features required
        geo_norm: Normalization factor for geographic distance (default 20000 km)
        
    Returns:
        Average combined quantization error
    """
    n_samples = data.shape[0]
    total_error = 0.0
    valid_count = 0
    
    for i in prange(n_samples):
        sample = data[i]
        coord = coordinates[i]
        
        # Find BMU with combined distance
        min_dist = np.inf
        for j in range(weights.shape[0]):
            # Feature distance
            feat_dist = bray_curtis_distance_single(sample, weights[j], min_valid_features)
            
            if not np.isinf(feat_dist):
                # Geographic distance (if using spatial weight)
                if spatial_weight > 0:
                    geo_dist = haversine_distance_single(
                        coord[0], coord[1],  # lon1, lat1
                        neuron_coords[j, 0], neuron_coords[j, 1]  # lon2, lat2
                    )
                    geo_dist_norm = geo_dist / geo_norm
                    
                    # Combined distance
                    combined = (1 - spatial_weight) * feat_dist + spatial_weight * geo_dist_norm
                else:
                    combined = feat_dist
                
                if combined < min_dist:
                    min_dist = combined
        
        # Accumulate valid errors
        if not np.isinf(min_dist):
            total_error += min_dist
            valid_count += 1
    
    return total_error / valid_count if valid_count > 0 else np.inf


# ============================================================================
# TOPOGRAPHIC ERROR - Direct calculation
# ============================================================================

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def calculate_te_numba(data, weights, grid_positions, min_valid_features=2):
    """Calculate Topographic Error directly in parallel.
    
    Measures how often first and second BMUs are not neighbors.
    
    Args:
        data: Input data (n_samples, n_features)
        weights: SOM weights (n_neurons, n_features)
        grid_positions: 2D grid positions of neurons (n_neurons, 2)
        min_valid_features: Minimum valid features required
        
    Returns:
        Topographic error (fraction of non-adjacent BMU pairs)
    """
    n_samples = data.shape[0]
    te_errors = 0
    valid_count = 0
    
    for i in prange(n_samples):
        sample = data[i]
        
        # Find first and second BMUs
        bmu1_idx, bmu2_idx = find_two_bmus_numba(sample, weights, min_valid_features)
        
        if bmu1_idx >= 0 and bmu2_idx >= 0:
            # Check if BMUs are neighbors on the grid
            pos1 = grid_positions[bmu1_idx]
            pos2 = grid_positions[bmu2_idx]
            
            # Grid distance (Manhattan distance)
            grid_dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
            
            # Neighbors have grid distance of 1
            if grid_dist > 1:
                te_errors += 1
            valid_count += 1
    
    return float(te_errors) / valid_count if valid_count > 0 else 0.0


# ============================================================================
# GEOGRAPHIC COHERENCE - Spatial pattern analysis
# ============================================================================

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def calculate_geographic_coherence_numba(data, weights, coordinates, min_valid_features=2,
                                        distance_threshold=50.0):
    """Calculate geographic coherence using simplified Moran's I.
    
    Measures spatial autocorrelation of BMU assignments.
    
    Args:
        data: Input data (n_samples, n_features)
        weights: SOM weights (n_neurons, n_features)
        coordinates: Sample coordinates (n_samples, 2) [lon, lat]
        min_valid_features: Minimum valid features required
        distance_threshold: Distance threshold for neighbors in km (default 50 km)
        
    Returns:
        Coherence score (-1 to 1, higher means more coherent)
    """
    n_samples = data.shape[0]
    
    # Find BMUs for all samples
    bmus = np.zeros(n_samples, dtype=np.int32)
    for i in prange(n_samples):
        bmus[i] = find_bmu_numba(data[i], weights, min_valid_features)
    
    # Calculate mean BMU index
    valid_bmus = bmus[bmus >= 0]
    if len(valid_bmus) == 0:
        return 0.0
    
    mean_bmu = np.mean(valid_bmus)
    
    # Calculate spatial autocorrelation
    numerator = 0.0
    denominator = 0.0
    weight_sum = 0.0
    
    for i in range(n_samples):
        if bmus[i] < 0:
            continue
            
        for j in range(i + 1, n_samples):
            if bmus[j] < 0:
                continue
            
            # Geographic distance
            geo_dist = haversine_distance_single(
                coordinates[i, 0], coordinates[i, 1],
                coordinates[j, 0], coordinates[j, 1]
            )
            
            # Spatial weight (1 if neighbors, 0 otherwise)
            if geo_dist <= distance_threshold:
                w_ij = 1.0
                weight_sum += w_ij
                
                # Deviation products
                dev_i = bmus[i] - mean_bmu
                dev_j = bmus[j] - mean_bmu
                numerator += w_ij * dev_i * dev_j
    
    # Variance of BMUs
    for i in range(n_samples):
        if bmus[i] >= 0:
            denominator += (bmus[i] - mean_bmu) ** 2
    
    if weight_sum > 0 and denominator > 0:
        morans_i = (n_samples / weight_sum) * (numerator / denominator)
        # Normalize to [-1, 1] range
        if morans_i < -1.0:
            return -1.0
        elif morans_i > 1.0:
            return 1.0
        else:
            return morans_i
    
    return 0.0


# ============================================================================
# BATCH PREDICTION - For efficiency
# ============================================================================

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def predict_batch_numba(data, weights, min_valid_features=2):
    """Predict BMUs for a batch of samples.
    
    Args:
        data: Input data (n_samples, n_features)
        weights: SOM weights (n_neurons, n_features)
        min_valid_features: Minimum valid features required
        
    Returns:
        BMU indices for each sample (n_samples,)
    """
    n_samples = data.shape[0]
    bmus = np.zeros(n_samples, dtype=np.int32)
    
    for i in prange(n_samples):
        bmus[i] = find_bmu_numba(data[i], weights, min_valid_features)
    
    return bmus