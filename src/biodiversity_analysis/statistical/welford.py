"""
Welford's algorithm for numerically stable incremental statistics.

"Get your math right or get out."
No approximations, no bullshit.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class WelfordAccumulator:
    """
    Incremental calculation of mean, variance, and covariance.
    
    This is the CORRECT way to do streaming statistics.
    Not that garbage you had before.
    """
    
    def __init__(self, n_features: int):
        """Initialize accumulator for n features."""
        self.n = 0
        self.mean = np.zeros(n_features)
        self.M2 = np.zeros(n_features)  # Sum of squares of differences from mean
        self.cov_sum = np.zeros((n_features, n_features))  # For covariance
        self.n_features = n_features
    
    def update(self, values: np.ndarray):
        """
        Update statistics with new sample(s).
        
        Args:
            values: Either single sample (n_features,) or batch (n_samples, n_features)
        """
        if values.ndim == 1:
            values = values.reshape(1, -1)
        
        for sample in values:
            # Skip NaN samples
            if np.any(np.isnan(sample)):
                continue
            
            self.n += 1
            delta = sample - self.mean
            self.mean += delta / self.n
            delta2 = sample - self.mean
            
            # Update variance terms
            self.M2 += delta * delta2
            
            # Update covariance terms
            for i in range(self.n_features):
                for j in range(i, self.n_features):
                    self.cov_sum[i, j] += delta[i] * delta2[j]
                    if i != j:
                        self.cov_sum[j, i] = self.cov_sum[i, j]
    
    def get_statistics(self) -> Dict[str, np.ndarray]:
        """Get current statistics."""
        if self.n < 2:
            return {
                'mean': self.mean,
                'variance': np.zeros(self.n_features),
                'std': np.zeros(self.n_features),
                'correlation': np.eye(self.n_features),
                'n': self.n
            }
        
        variance = self.M2 / (self.n - 1)
        std = np.sqrt(variance)
        
        # Correlation matrix
        corr = np.eye(self.n_features)
        for i in range(self.n_features):
            for j in range(i+1, self.n_features):
                if std[i] > 0 and std[j] > 0:
                    cov_ij = self.cov_sum[i, j] / (self.n - 1)
                    corr[i, j] = cov_ij / (std[i] * std[j])
                    corr[j, i] = corr[i, j]
        
        return {
            'mean': self.mean,
            'variance': variance,
            'std': std,
            'correlation': corr,
            'n': self.n
        }
    
    def merge(self, other: 'WelfordAccumulator'):
        """
        Merge another accumulator (for parallel processing).
        
        Chan's parallel algorithm for combining statistics.
        """
        if other.n == 0:
            return
        
        n_total = self.n + other.n
        delta = other.mean - self.mean
        
        # Merge means
        self.mean = (self.n * self.mean + other.n * other.mean) / n_total
        
        # Merge M2 (for variance)
        self.M2 = self.M2 + other.M2 + delta**2 * self.n * other.n / n_total
        
        # Merge covariance terms (Chan's formula)
        for i in range(self.n_features):
            for j in range(i, self.n_features):
                delta_i = other.mean[i] - self.mean[i]
                delta_j = other.mean[j] - self.mean[j]
                self.cov_sum[i, j] = (self.cov_sum[i, j] + other.cov_sum[i, j] + 
                                     delta_i * delta_j * self.n * other.n / n_total)
                if i != j:
                    self.cov_sum[j, i] = self.cov_sum[i, j]
        
        self.n = n_total


def chunked_correlation(rasters: Dict[str, np.ndarray], 
                        chunk_size: int = 100000) -> Dict[str, any]:
    """
    CORRECT chunked correlation using Welford's algorithm.
    
    Args:
        rasters: Dict of name -> 2D array
        chunk_size: Pixels per chunk
        
    Returns:
        Correlation matrix and statistics
    """
    names = list(rasters.keys())
    n_features = len(names)
    accumulator = WelfordAccumulator(n_features)
    
    # Get dimensions
    shape = next(iter(rasters.values())).shape
    height, width = shape
    total_pixels = height * width
    
    # Process in chunks
    for start_idx in range(0, total_pixels, chunk_size):
        end_idx = min(start_idx + chunk_size, total_pixels)
        
        # Extract chunk for each feature
        chunk_data = np.zeros((end_idx - start_idx, n_features))
        for i, name in enumerate(names):
            flat = rasters[name].flatten()
            chunk_data[:, i] = flat[start_idx:end_idx]
        
        # Update statistics
        accumulator.update(chunk_data)
    
    # Get final statistics
    stats = accumulator.get_statistics()
    
    # Format as correlation dict
    corr_matrix = stats['correlation']
    correlations = {}
    for i, name_i in enumerate(names):
        correlations[name_i] = {}
        for j, name_j in enumerate(names):
            correlations[name_i][name_j] = float(corr_matrix[i, j])
    
    return {
        'correlations': correlations,
        'statistics': {
            'n_samples': stats['n'],
            'means': {name: float(stats['mean'][i]) for i, name in enumerate(names)},
            'stds': {name: float(stats['std'][i]) for i, name in enumerate(names)},
            'method': 'welford_incremental'
        }
    }