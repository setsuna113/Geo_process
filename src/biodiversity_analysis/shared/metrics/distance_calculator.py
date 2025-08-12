"""Shared distance calculations for biodiversity analysis.

This module provides optimized distance calculations that can be used
by all analysis methods (SOM, K-means, etc.) ensuring consistency.
"""

import numpy as np
from numba import jit  # Removed prange to prevent thread explosion
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@jit(nopython=True, parallel=False, cache=True, fastmath=True)  # parallel=False to prevent thread explosion
def bray_curtis_distances_numba(data: np.ndarray, weights: np.ndarray, 
                                min_valid_features: int = 2) -> np.ndarray:
    """Numba-optimized partial Bray-Curtis distance calculation.
    
    Computes Bray-Curtis distances between all data samples and all weight vectors,
    handling missing values (NaN) by only considering valid feature pairs.
    
    Formula: BC(u,v) = Σ|u_i - v_i| / Σ(u_i + v_i)
    
    Args:
        data: Input data (n_samples, n_features), may contain NaN
        weights: Weight vectors (n_neurons, n_features)
        min_valid_features: Minimum valid features required for valid distance
        
    Returns:
        Distance matrix (n_samples, n_neurons), inf for invalid comparisons
    """
    n_samples = data.shape[0]
    n_neurons = weights.shape[0]
    n_features = data.shape[1]
    distances = np.zeros((n_samples, n_neurons), dtype=np.float32)
    
    for i in range(n_samples):  # Changed from prange to prevent thread explosion
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
    
    Handles missing values by ignoring NaN pairs.
    
    Args:
        data: Input data (n_samples, n_features), may contain NaN
        weights: Weight vectors (n_neurons, n_features)
        
    Returns:
        Squared distance matrix (n_samples, n_neurons)
    """
    n_samples = data.shape[0]
    n_neurons = weights.shape[0]
    n_features = data.shape[1]
    distances = np.zeros((n_samples, n_neurons), dtype=np.float32)
    
    for i in range(n_samples):  # Changed from prange to prevent thread explosion
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


@jit(nopython=True)
def haversine_distance_numba(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Numba-optimized haversine distance calculation.
    
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
    
    # Earth radius in km
    r = 6371.0
    return c * r


class SharedDistanceCalculator:
    """Shared distance calculator for all biodiversity analysis methods.
    
    This ensures consistency across different analysis methods and avoids
    code duplication.
    """
    
    def __init__(self, distance_metric: str = 'bray_curtis', 
                 min_valid_features: int = 2):
        """Initialize shared distance calculator.
        
        Args:
            distance_metric: 'bray_curtis' or 'euclidean'
            min_valid_features: Minimum valid features for distance calculation
        """
        self.distance_metric = distance_metric
        self.min_valid_features = min_valid_features
        logger.info(f"SharedDistanceCalculator initialized with {distance_metric} metric")
    
    def compute_distances(self, data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute distance matrix between data and weights.
        
        Args:
            data: Input samples (n_samples, n_features)
            weights: Weight vectors (n_weights, n_features)
            
        Returns:
            Distance matrix (n_samples, n_weights)
        """
        # Ensure contiguous memory layout for optimal performance
        data = np.ascontiguousarray(data, dtype=np.float32)
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        
        if self.distance_metric == 'bray_curtis':
            return bray_curtis_distances_numba(data, weights, self.min_valid_features)
        else:
            return euclidean_distances_numba(data, weights)
    
    def compute_pairwise_distances(self, coords1: np.ndarray, coords2: np.ndarray, 
                                  metric: str = 'haversine') -> np.ndarray:
        """Compute pairwise distances between coordinate sets.
        
        Args:
            coords1: First set of coordinates (n_samples1, 2) [lat, lon]
            coords2: Second set of coordinates (n_samples2, 2) [lat, lon]
            metric: Distance metric ('haversine' for geographic)
            
        Returns:
            Distance matrix (n_samples1, n_samples2)
        """
        if metric == 'haversine':
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
        else:
            # Use euclidean for non-geographic distances
            return euclidean_distances_numba(coords1, coords2)
    
    def find_bmus(self, data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Find Best Matching Units (BMUs) for each data sample.
        
        Args:
            data: Input samples (n_samples, n_features)
            weights: Weight vectors (n_weights, n_features)
            
        Returns:
            BMU indices for each sample (n_samples,)
        """
        distances = self.compute_distances(data, weights)
        return np.argmin(distances, axis=1)