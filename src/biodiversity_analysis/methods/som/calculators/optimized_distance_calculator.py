"""Optimized distance calculations for SOM training.

This module provides Numba JIT-compiled distance calculations for maximum performance.
All optimizations are built-in with no configuration options to avoid redundancy.
"""

import numpy as np
from numba import jit, prange
import logging

logger = logging.getLogger(__name__)


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def bray_curtis_distances_numba(data: np.ndarray, weights: np.ndarray, 
                                min_valid_features: int = 2) -> np.ndarray:
    """Numba-optimized partial Bray-Curtis distance calculation.
    
    Computes Bray-Curtis distances between all data samples and all weight vectors,
    handling missing values (NaN) by only considering valid feature pairs.
    
    Formula: BC(u,v) = Σ|u_i - v_i| / Σ(u_i + v_i)
    
    Args:
        data: Input data (n_samples, n_features), may contain NaN
        weights: SOM weights (n_neurons, n_features)
        min_valid_features: Minimum valid features required for valid distance
        
    Returns:
        Distance matrix (n_samples, n_neurons), inf for invalid comparisons
    """
    n_samples = data.shape[0]
    n_neurons = weights.shape[0]
    n_features = data.shape[1]
    distances = np.zeros((n_samples, n_neurons), dtype=np.float32)
    
    for i in prange(n_samples):
        for j in range(n_neurons):
            numerator = 0.0
            denominator = 0.0
            valid_count = 0
            
            for k in range(n_features):
                data_val = data[i, k]
                weight_val = weights[j, k]
                
                # Check if both values are valid (not NaN)
                if not np.isnan(data_val) and not np.isnan(weight_val):
                    abs_diff = abs(data_val - weight_val)
                    sum_val = data_val + weight_val
                    
                    if sum_val > 0:  # Avoid division by zero
                        numerator += abs_diff
                        denominator += sum_val
                        valid_count += 1
            
            # Calculate distance if enough valid features
            if valid_count >= min_valid_features and denominator > 0:
                distances[i, j] = numerator / denominator
            else:
                distances[i, j] = np.inf
                
    return distances


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def euclidean_distances_numba(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Numba-optimized Euclidean distance calculation.
    
    Fallback for when Bray-Curtis is not appropriate.
    Handles missing values by ignoring NaN pairs.
    
    Args:
        data: Input data (n_samples, n_features), may contain NaN
        weights: SOM weights (n_neurons, n_features)
        
    Returns:
        Squared distance matrix (n_samples, n_neurons)
    """
    n_samples = data.shape[0]
    n_neurons = weights.shape[0]
    n_features = data.shape[1]
    distances = np.zeros((n_samples, n_neurons), dtype=np.float32)
    
    for i in prange(n_samples):
        for j in range(n_neurons):
            dist = 0.0
            valid_count = 0
            
            for k in range(n_features):
                data_val = data[i, k]
                weight_val = weights[j, k]
                
                if not np.isnan(data_val) and not np.isnan(weight_val):
                    diff = data_val - weight_val
                    dist += diff * diff
                    valid_count += 1
            
            if valid_count > 0:
                distances[i, j] = dist
            else:
                distances[i, j] = np.inf
                
    return distances


@jit(nopython=True, cache=True, fastmath=True)
def find_bmu_indices(distances: np.ndarray) -> np.ndarray:
    """Find Best Matching Unit indices from distance matrix.
    
    Args:
        distances: Distance matrix (n_samples, n_neurons)
        
    Returns:
        BMU indices for each sample (n_samples,)
    """
    n_samples = distances.shape[0]
    bmu_indices = np.zeros(n_samples, dtype=np.int32)
    
    for i in range(n_samples):
        min_dist = np.inf
        min_idx = -1
        
        for j in range(distances.shape[1]):
            if distances[i, j] < min_dist:
                min_dist = distances[i, j]
                min_idx = j
        
        bmu_indices[i] = min_idx
        
    return bmu_indices


class OptimizedDistanceCalculator:
    """High-performance distance calculator for SOM training.
    
    Uses Numba JIT compilation with all optimizations enabled by default.
    No configuration options to avoid redundancy - always uses best settings.
    """
    
    def __init__(self, distance_metric: str = 'bray_curtis', 
                 min_valid_features: int = 2):
        """Initialize optimized distance calculator.
        
        Args:
            distance_metric: 'bray_curtis' or 'euclidean'
            min_valid_features: Minimum valid features for Bray-Curtis
        """
        self.distance_metric = distance_metric
        self.min_valid_features = min_valid_features
        
        # Select appropriate distance function
        if distance_metric == 'bray_curtis':
            self.distance_func = self._bray_curtis_distances
        else:
            self.distance_func = self._euclidean_distances
            
        logger.info(f"OptimizedDistanceCalculator initialized with {distance_metric} metric")
        
    def _bray_curtis_distances(self, data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Wrapper for Numba-optimized Bray-Curtis calculation."""
        return bray_curtis_distances_numba(data, weights, self.min_valid_features)
    
    def _euclidean_distances(self, data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Wrapper for Numba-optimized Euclidean calculation."""
        return euclidean_distances_numba(data, weights)
    
    def compute_distances(self, data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute distance matrix between data and weights.
        
        Args:
            data: Input samples (n_samples, n_features)
            weights: SOM weights (n_neurons, n_features)
            
        Returns:
            Distance matrix (n_samples, n_neurons)
        """
        # Ensure contiguous memory layout for optimal performance
        data = np.ascontiguousarray(data, dtype=np.float32)
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        
        return self.distance_func(data, weights)
    
    def find_bmus(self, data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Find Best Matching Units for data samples.
        
        Args:
            data: Input samples (n_samples, n_features)
            weights: SOM weights (n_neurons, n_features)
            
        Returns:
            BMU indices (n_samples,)
        """
        distances = self.compute_distances(data, weights)
        return find_bmu_indices(distances)