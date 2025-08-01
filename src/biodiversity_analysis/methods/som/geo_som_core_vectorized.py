"""Vectorized implementation of batch update for GeoSOM."""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def vectorized_partial_bray_curtis(data: np.ndarray, weights: np.ndarray, 
                                   min_valid_features: int = 2) -> np.ndarray:
    """Vectorized Bray-Curtis distance for data with missing values.
    
    Args:
        data: Input samples (n_samples, n_features)
        weights: Weight vectors (n_neurons, n_features)
        min_valid_features: Minimum valid features for comparison
        
    Returns:
        Distance matrix (n_samples, n_neurons)
    """
    n_samples = data.shape[0]
    n_neurons = weights.shape[0]
    
    # Expand dimensions for broadcasting
    data_exp = data[:, np.newaxis, :]  # (n_samples, 1, n_features)
    weights_exp = weights[np.newaxis, :, :]  # (1, n_neurons, n_features)
    
    # Find valid pairs (not NaN in both data and weights)
    valid_mask = ~(np.isnan(data_exp) | np.isnan(weights_exp))
    n_valid = valid_mask.sum(axis=2)  # (n_samples, n_neurons)
    
    # Replace NaN with 0 for calculation (will be masked out)
    data_clean = np.nan_to_num(data_exp, nan=0.0)
    weights_clean = np.nan_to_num(weights_exp, nan=0.0)
    
    # Vectorized Bray-Curtis calculation
    diff = np.abs(data_clean - weights_clean)
    sum_vals = data_clean + weights_clean
    
    # Apply mask and sum
    numerator = (diff * valid_mask).sum(axis=2)
    denominator = (sum_vals * valid_mask).sum(axis=2)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        distances = numerator / denominator
    
    # Set invalid distances (insufficient features or zero denominator)
    invalid = (n_valid < min_valid_features) | (denominator == 0)
    distances[invalid] = np.inf
    
    return distances


def vectorized_find_bmus(distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find best matching units for all samples at once.
    
    Args:
        distances: Distance matrix (n_samples, n_neurons)
        
    Returns:
        bmu_indices: BMU index for each sample
        min_distances: Minimum distance for each sample
    """
    bmu_indices = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)
    
    # Mark invalid BMUs (where all distances are inf)
    invalid_mask = np.all(np.isinf(distances), axis=1)
    bmu_indices[invalid_mask] = -1  # INVALID_INDEX
    
    return bmu_indices, min_distances


def vectorized_neighborhood(bmu_indices: np.ndarray, grid_positions: np.ndarray,
                           current_radius: float, n_neurons: int) -> np.ndarray:
    """Calculate neighborhood influences for all BMUs at once.
    
    Args:
        bmu_indices: BMU index for each sample
        grid_positions: Neuron positions on grid (n_neurons, 2)
        current_radius: Current neighborhood radius
        n_neurons: Total number of neurons
        
    Returns:
        influences: Neighborhood influence matrix (n_samples, n_neurons)
    """
    n_samples = len(bmu_indices)
    influences = np.zeros((n_samples, n_neurons))
    
    # Handle valid BMUs
    valid_mask = bmu_indices >= 0
    valid_bmus = bmu_indices[valid_mask]
    
    if len(valid_bmus) > 0:
        # Get BMU positions
        bmu_positions = grid_positions[valid_bmus]  # (n_valid, 2)
        
        # Calculate distances from each BMU to all neurons
        # Using broadcasting: (n_valid, 1, 2) - (1, n_neurons, 2)
        pos_diff = bmu_positions[:, np.newaxis, :] - grid_positions[np.newaxis, :, :]
        distances_sq = np.sum(pos_diff ** 2, axis=2)  # (n_valid, n_neurons)
        
        # Gaussian neighborhood
        neighborhood = np.exp(-distances_sq / (2 * current_radius ** 2))
        
        # Assign influences
        influences[valid_mask] = neighborhood
    
    return influences


def vectorized_batch_update(data: np.ndarray, weights: np.ndarray,
                           grid_positions: np.ndarray,
                           current_lr: float, current_radius: float,
                           min_valid_features: int = 2,
                           coordinates: Optional[np.ndarray] = None) -> np.ndarray:
    """Fully vectorized batch update for SOM weights.
    
    Args:
        data: Input samples (n_samples, n_features)
        weights: Current weights (n_neurons, n_features)
        grid_positions: Neuron grid positions (n_neurons, 2)
        current_lr: Current learning rate
        current_radius: Current neighborhood radius
        min_valid_features: Minimum valid features for distance
        coordinates: Geographic coordinates (optional)
        
    Returns:
        Updated weights
    """
    n_samples, n_features = data.shape
    n_neurons = weights.shape[0]
    
    # Step 1: Calculate all distances at once
    distances = vectorized_partial_bray_curtis(data, weights, min_valid_features)
    
    # Step 2: Find all BMUs
    bmu_indices, _ = vectorized_find_bmus(distances)
    
    # Step 3: Calculate neighborhood influences for all samples
    influences = vectorized_neighborhood(bmu_indices, grid_positions, 
                                       current_radius, n_neurons)
    
    # Step 4: Accumulate updates using matrix operations
    # For each neuron, sum weighted samples
    numerator = np.zeros_like(weights)
    denominator = np.zeros(n_neurons)
    
    # Vectorized accumulation
    for j in range(n_neurons):
        # Get influence of neuron j for all samples
        neuron_influences = influences[:, j]  # (n_samples,)
        
        if np.any(neuron_influences > 0):
            # Weighted sum of samples
            # Handle NaN by creating a mask
            valid_mask = ~np.isnan(data)
            
            # Broadcast influence weights
            weighted_data = data * neuron_influences[:, np.newaxis]
            
            # Sum only valid values
            numerator[j] = np.nansum(weighted_data, axis=0)
            
            # Count contributions per feature
            contributions = (valid_mask * neuron_influences[:, np.newaxis])
            denominator[j] = np.sum(contributions.any(axis=0))
    
    # Step 5: Apply updates
    # Only update neurons with sufficient data
    update_mask = denominator > 1
    
    if np.any(update_mask):
        # Calculate targets
        targets = np.zeros_like(weights)
        targets[update_mask] = numerator[update_mask] / denominator[update_mask, np.newaxis]
        
        # Adaptive momentum
        momentum = np.minimum(denominator / n_samples, 1.0)
        effective_lr = momentum * current_lr * 0.5
        
        # Apply updates with NaN handling
        for j in np.where(update_mask)[0]:
            valid = ~(np.isnan(targets[j]) | np.isnan(weights[j]))
            weights[j, valid] = (1 - effective_lr[j]) * weights[j, valid] + \
                               effective_lr[j] * targets[j, valid]
    
    return weights


# Optimized version using einsum for even better performance
def vectorized_batch_update_optimized(data: np.ndarray, weights: np.ndarray,
                                     grid_positions: np.ndarray,
                                     current_lr: float, current_radius: float,
                                     min_valid_features: int = 2) -> np.ndarray:
    """Ultra-optimized vectorized batch update using einsum.
    
    This version minimizes loops and uses einsum for efficient tensor operations.
    """
    n_samples, n_features = data.shape
    n_neurons = weights.shape[0]
    
    # Calculate distances
    distances = vectorized_partial_bray_curtis(data, weights, min_valid_features)
    
    # Find BMUs
    bmu_indices, _ = vectorized_find_bmus(distances)
    
    # Calculate influences
    influences = vectorized_neighborhood(bmu_indices, grid_positions, 
                                       current_radius, n_neurons)
    
    # Vectorized accumulation using einsum
    # Replace NaN with 0 and create validity mask
    data_clean = np.nan_to_num(data, nan=0.0)
    valid_mask = ~np.isnan(data)
    
    # Weighted sum: sum over samples dimension
    # numerator[j,k] = sum_i influences[i,j] * data[i,k] * valid[i,k]
    numerator = np.einsum('ij,ik,ik->jk', influences, data_clean, valid_mask)
    
    # Denominator: sum of influences where data is valid
    # denominator[j] = sum_i,k influences[i,j] * valid[i,k]
    denominator = np.einsum('ij,ik->j', influences, valid_mask)
    
    # Update only neurons with sufficient data
    update_mask = denominator > n_features  # At least one full sample worth
    
    # Calculate new weights
    new_weights = weights.copy()
    
    if np.any(update_mask):
        # Safe division
        targets = np.zeros_like(weights)
        targets[update_mask] = numerator[update_mask] / denominator[update_mask, np.newaxis]
        
        # Adaptive learning rate
        momentum = np.minimum(denominator / (n_samples * n_features), 1.0)
        effective_lr = momentum * current_lr * 0.5
        
        # Vectorized update
        blend = effective_lr[update_mask, np.newaxis]
        valid = ~(np.isnan(targets[update_mask]) | np.isnan(weights[update_mask]))
        
        new_weights[update_mask] = np.where(
            valid,
            (1 - blend) * weights[update_mask] + blend * targets[update_mask],
            weights[update_mask]
        )
    
    return new_weights


def benchmark_implementations(n_samples=10000, n_features=4, n_neurons=225):
    """Benchmark vectorized vs sequential implementations."""
    import time
    
    # Generate test data
    data = np.random.rand(n_samples, n_features)
    data[np.random.rand(n_samples, n_features) > 0.5] = np.nan  # 50% missing
    weights = np.random.rand(n_neurons, n_features)
    grid_positions = np.array([[i, j] for i in range(15) for j in range(15)])
    
    # Time vectorized version
    start = time.time()
    new_weights = vectorized_batch_update_optimized(
        data, weights, grid_positions, 0.5, 2.0
    )
    vectorized_time = time.time() - start
    
    print(f"Vectorized implementation: {vectorized_time:.3f} seconds")
    print(f"Estimated time for 9M samples: {vectorized_time * 900:.1f} seconds")
    
    return new_weights