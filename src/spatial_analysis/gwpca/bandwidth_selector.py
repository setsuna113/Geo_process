# src/spatial_analysis/gwpca/bandwidth_selector.py
"""Bandwidth selection methods for GWPCA."""

import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

class BandwidthSelector:
    """Select optimal bandwidth for geographically weighted methods."""
    
    def __init__(self, coords: np.ndarray, kernel: str = 'bisquare'):
        """
        Initialize bandwidth selector.
        
        Args:
            coords: Coordinate array (n_points, 2)
            kernel: Kernel type
        """
        self.coords = coords
        self.kernel = kernel
        self.n_points = len(coords)
        
        # Precompute distance matrix for efficiency
        self.distances = cdist(coords, coords)
    
    def golden_section_search(self, 
                            objective_func,
                            lower: float,
                            upper: float,
                            tolerance: float = 1e-4) -> float:
        """
        Golden section search for optimal bandwidth.
        
        Args:
            objective_func: Function to minimize
            lower: Lower bound
            upper: Upper bound
            tolerance: Convergence tolerance
            
        Returns:
            Optimal bandwidth
        """
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        while upper - lower > tolerance:
            x1 = upper - (upper - lower) / golden_ratio
            x2 = lower + (upper - lower) / golden_ratio
            
            if objective_func(x1) < objective_func(x2):
                upper = x2
            else:
                lower = x1
        
        return (upper + lower) / 2
    
    def cv_score(self, bandwidth: float, X: np.ndarray, 
                adaptive: bool = True,
                n_folds: int = 5) -> float:
        """
        Cross-validation score for bandwidth.
        
        Args:
            bandwidth: Bandwidth to test
            X: Data matrix
            adaptive: Whether bandwidth is adaptive
            n_folds: Number of CV folds
            
        Returns:
            CV score (lower is better)
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, test_idx in kf.split(X):
            # Compute weights for test points
            test_weights = self._compute_weights(
                test_idx, bandwidth, adaptive
            )
            
            # Predict and score
            predictions = self._weighted_predict(
                X[train_idx], X[test_idx], test_weights
            )
            
            # MSE
            mse = np.mean((X[test_idx] - predictions) ** 2)
            scores.append(mse)
        
        return np.mean(scores)
    
    def aic_score(self, bandwidth: float, X: np.ndarray,
                 adaptive: bool = True) -> float:
        """
        AIC score for bandwidth selection.
        
        Args:
            bandwidth: Bandwidth to test
            X: Data matrix
            adaptive: Whether bandwidth is adaptive
            
        Returns:
            AIC score (lower is better)
        """
        n = len(X)
        
        # Compute effective number of parameters
        trace_s = self._compute_trace_s(bandwidth, adaptive)
        
        # Compute RSS
        rss = 0
        for i in range(n):
            weights = self._compute_weights([i], bandwidth, adaptive)[0]
            pred = self._weighted_predict(X, X[i:i+1], [weights])[0]
            rss += np.sum((X[i] - pred) ** 2)
        
        # AIC = n * log(RSS/n) + 2 * trace(S)
        aic = n * np.log(rss / n) + 2 * trace_s
        
        return aic
    
    def _compute_weights(self, indices: List[int], 
                       bandwidth: float,
                       adaptive: bool) -> List[np.ndarray]:
        """Compute spatial weights for given indices."""
        weights_list = []
        
        for idx in indices:
            if adaptive:
                # Adaptive: bandwidth is number of neighbors
                sorted_dists = np.sort(self.distances[idx])
                if int(bandwidth) < len(sorted_dists):
                    bw_dist = sorted_dists[int(bandwidth)]
                else:
                    bw_dist = sorted_dists[-1]
                
                weights = self._kernel_function(
                    self.distances[idx], bw_dist
                )
            else:
                # Fixed bandwidth
                weights = self._kernel_function(
                    self.distances[idx], bandwidth
                )
            
            weights_list.append(weights)
        
        return weights_list
    
    def _kernel_function(self, distances: np.ndarray, 
                       bandwidth: float) -> np.ndarray:
        """Apply kernel function to distances."""
        if self.kernel == 'gaussian':
            return np.exp(-(distances / bandwidth) ** 2)
        elif self.kernel == 'bisquare':
            u = distances / bandwidth
            return np.where(u <= 1, (1 - u**2)**2, 0)
        elif self.kernel == 'exponential':
            return np.exp(-distances / bandwidth)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _weighted_predict(self, X_train: np.ndarray, 
                        X_test: np.ndarray,
                        weights_list: List[np.ndarray]) -> np.ndarray:
        """Make weighted predictions."""
        predictions = []
        
        for i, weights in enumerate(weights_list):
            # Weighted mean
            w_sum = np.sum(weights)
            if w_sum > 0:
                pred = np.sum(X_train * weights[:, np.newaxis], axis=0) / w_sum
            else:
                pred = np.mean(X_train, axis=0)
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _compute_trace_s(self, bandwidth: float, adaptive: bool) -> float:
        """Compute trace of hat matrix S."""
        trace = 0
        
        for i in range(self.n_points):
            weights = self._compute_weights([i], bandwidth, adaptive)[0]
            trace += weights[i] / np.sum(weights)
        
        return trace
    
    def suggest_bandwidth_range(self, adaptive: bool = True) -> Tuple[float, float]:
        """
        Suggest reasonable bandwidth range.
        
        Args:
            adaptive: Whether using adaptive bandwidth
            
        Returns:
            Tuple of (lower, upper) bounds
        """
        if adaptive:
            # For adaptive, use percentage of points
            lower = max(10, int(0.01 * self.n_points))
            upper = min(int(0.5 * self.n_points), 200)
        else:
            # For fixed, use percentage of max distance
            max_dist = np.max(self.distances)
            lower = 0.01 * max_dist
            upper = 0.3 * max_dist
        
        return lower, upper