"""Adaptive distance metrics for biodiversity k-means."""

import numpy as np
import logging
from numba import jit
from typing import Optional, Tuple
from .kmeans_config import KMeansConfig
from .weight_calculator import FeatureWeightCalculator

logger = logging.getLogger(__name__)


class AdaptivePartialDistance:
    """Implements weighted partial distance metrics with adaptive thresholds.
    
    This class handles:
    - Partial distance calculations for missing data
    - Feature weighting based on data quality
    - Adaptive minimum feature thresholds based on location
    """
    
    def __init__(self, config: KMeansConfig, feature_weights: Optional[np.ndarray] = None):
        """Initialize adaptive distance calculator.
        
        Args:
            config: K-means configuration
            feature_weights: Pre-calculated feature weights (optional)
        """
        self.config = config
        self.feature_weights = feature_weights
        self._distance_func = self._get_distance_function()
    
    def set_weights(self, weights: np.ndarray):
        """Update feature weights."""
        self.feature_weights = weights
    
    def _get_distance_function(self):
        """Get appropriate distance function based on config."""
        if self.config.distance_metric == 'bray_curtis':
            return weighted_partial_bray_curtis
        elif self.config.distance_metric == 'euclidean':
            return weighted_partial_euclidean
        else:
            raise ValueError(f"Unknown distance metric: {self.config.distance_metric}")
    
    def calculate_distance(self, u: np.ndarray, v: np.ndarray,
                          coord_u: Optional[np.ndarray] = None,
                          coord_v: Optional[np.ndarray] = None) -> float:
        """Calculate adaptive distance between two samples.
        
        Args:
            u, v: Feature vectors
            coord_u, coord_v: Geographic coordinates [lat, lon]
            
        Returns:
            distance: Adaptive weighted distance
        """
        # Get adaptive minimum features threshold
        min_features = self._get_min_features(coord_u, coord_v)
        
        # Calculate distance with adaptive threshold
        if self.feature_weights is not None:
            distance = self._distance_func(u, v, self.feature_weights, min_features)
        else:
            # Use equal weights if not provided
            weights = np.ones(len(u)) / len(u)
            distance = self._distance_func(u, v, weights, min_features)
        
        return distance
    
    def _get_min_features(self, coord_u: Optional[np.ndarray], 
                         coord_v: Optional[np.ndarray]) -> int:
        """Get minimum features threshold based on location.
        
        Args:
            coord_u, coord_v: Geographic coordinates [lat, lon]
            
        Returns:
            min_features: Adaptive threshold
        """
        if self.config.adaptive_mode == 'latitude' and coord_u is not None and coord_v is not None:
            # Use average latitude
            avg_lat = abs((coord_u[0] + coord_v[0]) / 2)
            
            if avg_lat > self.config.arctic_boundary:
                return self.config.arctic_min_features
            elif avg_lat > self.config.temperate_boundary:
                return self.config.temperate_min_features
            else:
                return self.config.tropical_min_features
        
        elif self.config.adaptive_mode == 'density':
            # TODO: Implement density-based thresholds
            logger.warning("Density-based thresholds not yet implemented")
            return self.config.temperate_min_features
        
        else:
            # Default to temperate threshold
            return self.config.temperate_min_features
    
    def pairwise_distances(self, X: np.ndarray, Y: Optional[np.ndarray] = None,
                          coords_X: Optional[np.ndarray] = None,
                          coords_Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate pairwise distances between samples.
        
        Args:
            X: First set of samples (n_samples_X, n_features)
            Y: Second set of samples (n_samples_Y, n_features), or None for X vs X
            coords_X, coords_Y: Geographic coordinates
            
        Returns:
            distances: Pairwise distance matrix
        """
        if Y is None:
            Y = X
            coords_Y = coords_X
        
        n_x = len(X)
        n_y = len(Y)
        distances = np.full((n_x, n_y), np.inf)
        
        for i in range(n_x):
            for j in range(n_y):
                coord_i = coords_X[i] if coords_X is not None else None
                coord_j = coords_Y[j] if coords_Y is not None else None
                
                distances[i, j] = self.calculate_distance(
                    X[i], Y[j], coord_i, coord_j
                )
        
        return distances


@jit(nopython=True)
def weighted_partial_bray_curtis(u: np.ndarray, v: np.ndarray, 
                                weights: np.ndarray, min_features: int) -> float:
    """Calculate weighted partial Bray-Curtis distance.
    
    Bray-Curtis: sum(|u-v|) / sum(u+v)
    Only uses non-missing features in both vectors.
    
    Args:
        u, v: Feature vectors
        weights: Feature weights
        min_features: Minimum valid features required
        
    Returns:
        distance: Weighted Bray-Curtis distance
    """
    valid_count = 0
    weighted_diff_sum = 0.0
    weighted_total_sum = 0.0
    
    for i in range(len(u)):
        if not (np.isnan(u[i]) or np.isnan(v[i])):
            valid_count += 1
            diff = abs(u[i] - v[i])
            total = u[i] + v[i]
            
            weighted_diff_sum += weights[i] * diff
            weighted_total_sum += weights[i] * total
    
    if valid_count < min_features:
        return np.inf
    
    if weighted_total_sum == 0:
        return 0.0 if weighted_diff_sum == 0 else np.inf
    
    return weighted_diff_sum / weighted_total_sum


@jit(nopython=True)
def weighted_partial_euclidean(u: np.ndarray, v: np.ndarray,
                              weights: np.ndarray, min_features: int) -> float:
    """Calculate weighted partial Euclidean distance.
    
    Only uses non-missing features in both vectors.
    
    Args:
        u, v: Feature vectors
        weights: Feature weights
        min_features: Minimum valid features required
        
    Returns:
        distance: Weighted Euclidean distance
    """
    valid_count = 0
    weighted_sq_sum = 0.0
    weight_sum = 0.0
    
    for i in range(len(u)):
        if not (np.isnan(u[i]) or np.isnan(v[i])):
            valid_count += 1
            diff = u[i] - v[i]
            weighted_sq_sum += weights[i] * diff * diff
            weight_sum += weights[i]
    
    if valid_count < min_features:
        return np.inf
    
    # Normalize by sum of weights used
    if weight_sum > 0:
        return np.sqrt(weighted_sq_sum / weight_sum) * np.sqrt(len(u) / valid_count)
    else:
        return np.inf