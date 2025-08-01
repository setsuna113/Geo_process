"""Partial distance metrics for handling missing data in biodiversity analysis.

Implements distance calculations that handle 70% missing values by only
comparing valid (non-NaN) pairs of features.
"""

import numpy as np
from typing import Optional, Callable
from numba import jit
from .constants import INVALID_DISTANCE, INVALID_INDEX


@jit(nopython=True)
def partial_bray_curtis_numba(u: np.ndarray, v: np.ndarray, min_valid: int = 2) -> float:
    """Numba-optimized partial Bray-Curtis distance.
    
    Args:
        u, v: Vectors to compare (may contain NaN)
        min_valid: Minimum number of valid pairs required
        
    Returns:
        Distance value or NaN if insufficient valid pairs
    """
    valid_count = 0
    numerator = 0.0
    denominator = 0.0
    
    for i in range(len(u)):
        if not (np.isnan(u[i]) or np.isnan(v[i])):
            valid_count += 1
            numerator += abs(u[i] - v[i])
            denominator += u[i] + v[i]
    
    if valid_count < min_valid:
        return INVALID_DISTANCE
    
    if denominator == 0.0:
        # Both vectors are zero
        return 0.0 if numerator == 0.0 else 1.0
    
    return numerator / denominator


@jit(nopython=True)
def partial_euclidean_numba(u: np.ndarray, v: np.ndarray, min_valid: int = 2) -> float:
    """Numba-optimized partial Euclidean distance.
    
    Args:
        u, v: Vectors to compare (may contain NaN)
        min_valid: Minimum number of valid pairs required
        
    Returns:
        Distance value or NaN if insufficient valid pairs
    """
    valid_count = 0
    sum_squared = 0.0
    
    for i in range(len(u)):
        if not (np.isnan(u[i]) or np.isnan(v[i])):
            valid_count += 1
            diff = u[i] - v[i]
            sum_squared += diff * diff
    
    if valid_count < min_valid:
        return INVALID_DISTANCE
    
    # Scale by proportion of valid features
    scale_factor = len(u) / valid_count
    return np.sqrt(sum_squared * scale_factor)


def partial_jaccard(u: np.ndarray, v: np.ndarray, min_valid: int = 2) -> float:
    """Partial Jaccard distance for presence/absence data.
    
    Args:
        u, v: Binary vectors (may contain NaN)
        min_valid: Minimum number of valid pairs required
        
    Returns:
        Distance value or NaN if insufficient valid pairs
    """
    valid = ~(np.isnan(u) | np.isnan(v))
    
    if valid.sum() < min_valid:
        return INVALID_DISTANCE
    
    u_valid = u[valid] > 0
    v_valid = v[valid] > 0
    
    intersection = np.sum(u_valid & v_valid)
    union = np.sum(u_valid | v_valid)
    
    if union == 0:
        return 0.0
    
    return 1.0 - (intersection / union)


def partial_cosine(u: np.ndarray, v: np.ndarray, min_valid: int = 2) -> float:
    """Partial cosine distance.
    
    Args:
        u, v: Vectors to compare (may contain NaN)
        min_valid: Minimum number of valid pairs required
        
    Returns:
        Distance value or NaN if insufficient valid pairs
    """
    valid = ~(np.isnan(u) | np.isnan(v))
    
    if valid.sum() < min_valid:
        return INVALID_DISTANCE
    
    u_valid = u[valid]
    v_valid = v[valid]
    
    norm_u = np.linalg.norm(u_valid)
    norm_v = np.linalg.norm(v_valid)
    
    if norm_u == 0 or norm_v == 0:
        return 1.0
    
    cosine_sim = np.dot(u_valid, v_valid) / (norm_u * norm_v)
    return 1.0 - cosine_sim


def get_partial_distance_function(metric: str, min_valid_features: int = 2) -> Callable:
    """Get partial distance function by name.
    
    Args:
        metric: Distance metric name
        min_valid_features: Minimum valid features required
        
    Returns:
        Distance function
    """
    metrics = {
        'bray_curtis': lambda u, v: partial_bray_curtis_numba(u, v, min_valid_features),
        'euclidean': lambda u, v: partial_euclidean_numba(u, v, min_valid_features),
        'jaccard': lambda u, v: partial_jaccard(u, v, min_valid_features),
        'cosine': lambda u, v: partial_cosine(u, v, min_valid_features)
    }
    
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    
    return metrics[metric]


class PartialDistanceMatrix:
    """Compute distance matrices with missing data handling."""
    
    def __init__(self, metric: str = 'bray_curtis', min_valid_features: int = 2):
        """Initialize with distance metric.
        
        Args:
            metric: Distance metric name
            min_valid_features: Minimum valid features for comparison
        """
        self.metric = metric
        self.min_valid_features = min_valid_features
        self.distance_func = get_partial_distance_function(metric, min_valid_features)
    
    def compute_pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute pairwise distance matrix.
        
        Args:
            X: First set of vectors (n_samples_1, n_features)
            Y: Second set of vectors (n_samples_2, n_features), or None for X vs X
            
        Returns:
            Distance matrix (n_samples_1, n_samples_2)
        """
        if Y is None:
            Y = X
            symmetric = True
        else:
            symmetric = False
        
        n_x = X.shape[0]
        n_y = Y.shape[0]
        distances = np.full((n_x, n_y), np.nan)
        
        for i in range(n_x):
            start_j = i if symmetric else 0
            for j in range(start_j, n_y):
                dist = self.distance_func(X[i], Y[j])
                distances[i, j] = dist
                if symmetric and i != j:
                    distances[j, i] = dist
        
        return distances
    
    def find_nearest(self, query: np.ndarray, references: np.ndarray) -> tuple:
        """Find nearest reference vector to query.
        
        Args:
            query: Query vector (n_features,)
            references: Reference vectors (n_references, n_features)
            
        Returns:
            Tuple of (index, distance) or (INVALID_INDEX, INVALID_DISTANCE) if no valid comparison
        """
        best_idx = INVALID_INDEX
        best_dist = np.inf
        
        for i, ref in enumerate(references):
            dist = self.distance_func(query, ref)
            if not np.isnan(dist) and dist < best_dist:
                best_dist = dist
                best_idx = i
        
        if best_idx == INVALID_INDEX:
            return INVALID_INDEX, INVALID_DISTANCE
        
        return best_idx, best_dist


def validate_missing_data_handling(data: np.ndarray, threshold: float = 0.7) -> dict:
    """Validate data for missing value handling.
    
    Args:
        data: Input data array
        threshold: Maximum allowed proportion of missing values
        
    Returns:
        Dictionary with validation results
    """
    n_samples, n_features = data.shape
    
    # Overall missing proportion
    missing_mask = np.isnan(data)
    overall_missing = missing_mask.sum() / data.size
    
    # Per-feature missing
    feature_missing = missing_mask.sum(axis=0) / n_samples
    
    # Per-sample missing
    sample_missing = missing_mask.sum(axis=1) / n_features
    
    # Find problematic features/samples
    problematic_features = np.where(feature_missing > threshold)[0]
    problematic_samples = np.where(sample_missing > threshold)[0]
    
    # Compute pairwise valid counts
    valid_pairs = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            valid = ~(missing_mask[i] | missing_mask[j])
            valid_pairs[i, j] = valid_pairs[j, i] = valid.sum()
    
    return {
        'overall_missing_proportion': overall_missing,
        'feature_missing_proportions': feature_missing,
        'sample_missing_proportions': sample_missing,
        'problematic_features': problematic_features,
        'problematic_samples': problematic_samples,
        'min_valid_pairs': valid_pairs.min(),
        'mean_valid_pairs': valid_pairs.mean(),
        'pairwise_valid_matrix': valid_pairs
    }