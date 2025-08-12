"""Optimized distance calculations for SOM training.

This module provides Numba JIT-compiled distance calculations for maximum performance.
All optimizations are built-in with no configuration options to avoid redundancy.
"""

import numpy as np
from numba import jit  # Removed prange to prevent thread explosion
import logging
from ..utils.spatial_utils import haversine_distance_numba

logger = logging.getLogger(__name__)


@jit(nopython=True, parallel=False, cache=True, fastmath=True)  # parallel=False to prevent thread explosion with joblib
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
    
    for i in range(n_samples):  # Changed from prange to prevent thread creation
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


@jit(nopython=True, parallel=False, cache=True, fastmath=True)  # parallel=False to prevent thread explosion
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
    
    for i in range(n_samples):  # Changed from prange to prevent thread creation
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
    
    def partial_bray_curtis_chunked(self, u: np.ndarray, v: np.ndarray, 
                                   min_valid_features: int = None) -> np.ndarray:
        """Compute partial Bray-Curtis distance between vectors.
        
        This method is for compatibility with the GeoSOM implementation.
        
        Args:
            u: First vector or array (n_samples1, n_features)
            v: Second vector or array (n_samples2, n_features)
            min_valid_features: Minimum valid features (uses instance default if None)
            
        Returns:
            Distance matrix (n_samples1, n_samples2)
        """
        if min_valid_features is None:
            min_valid_features = self.min_valid_features
            
        # Ensure 2D arrays
        if u.ndim == 1:
            u = u.reshape(1, -1)
        if v.ndim == 1:
            v = v.reshape(1, -1)
            
        # Use the optimized Bray-Curtis function
        return bray_curtis_distances_numba(u, v, min_valid_features)
    
    def combined_distance_chunked(self, data: np.ndarray, weights: np.ndarray,
                                 data_coords: np.ndarray, neuron_coords: np.ndarray,
                                 spatial_weight: float, min_valid_features: int = None) -> np.ndarray:
        """Compute combined feature and spatial distances.
        
        Args:
            data: Input data (n_samples, n_features)
            weights: SOM weights (n_neurons, n_features)
            data_coords: Coordinates for data samples (n_samples, 2) [x, y]
            neuron_coords: Coordinates for neurons (n_neurons, 2) [x, y]
            spatial_weight: Weight for spatial distance (0-1)
            min_valid_features: Minimum valid features required
            
        Returns:
            Combined distance matrix (n_samples, n_neurons)
        """
        if min_valid_features is None:
            min_valid_features = self.min_valid_features
            
        # Compute feature distances
        feature_distances = self.partial_bray_curtis_chunked(data, weights, min_valid_features)
        
        # Compute spatial distances
        spatial_distances = self.haversine_distances_chunked(data_coords, neuron_coords)
        
        # Normalize spatial distances to [0, 1] range
        max_spatial = np.max(spatial_distances)
        if max_spatial > 0:
            spatial_distances = spatial_distances / max_spatial
        
        # Combine distances
        combined_distances = (1 - spatial_weight) * feature_distances + spatial_weight * spatial_distances
        
        return combined_distances
    
    def haversine_distances_chunked(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """Compute haversine distances between two sets of coordinates.
        
        Args:
            coords1: First set of coordinates (n_samples1, 2) [lat, lon]
            coords2: Second set of coordinates (n_samples2, 2) [lat, lon]
            
        Returns:
            Distance matrix (n_samples1, n_samples2) in kilometers
        """
        n_samples1 = coords1.shape[0]
        n_samples2 = coords2.shape[0]
        distances = np.zeros((n_samples1, n_samples2), dtype=np.float32)
        
        for i in range(n_samples1):
            for j in range(n_samples2):
                # coords are [lat, lon] so lon=coords[:,1], lat=coords[:,0]
                distances[i, j] = haversine_distance_numba(
                    coords1[i, 1], coords1[i, 0],  # lon1, lat1
                    coords2[j, 1], coords2[j, 0]   # lon2, lat2
                )
        
        return distances