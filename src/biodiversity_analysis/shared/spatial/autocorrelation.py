"""
Spatial autocorrelation detection and handling.
"""

import numpy as np
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class SpatialAutocorrelation:
    """Detect and quantify spatial autocorrelation in data."""
    
    @staticmethod
    def morans_i(
        values: np.ndarray,
        coordinates: np.ndarray,
        weight_threshold: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate Moran's I statistic for spatial autocorrelation.
        
        Args:
            values: Data values at each location
            coordinates: Spatial coordinates (n, 2)
            weight_threshold: Distance threshold for weights (None = inverse distance)
            
        Returns:
            Tuple of (moran_i, p_value)
        """
        n = len(values)
        
        # Create spatial weights matrix
        if weight_threshold is None:
            # Inverse distance weights
            distances = np.sqrt(((coordinates[:, None] - coordinates) ** 2).sum(axis=2))
            np.fill_diagonal(distances, np.inf)  # Avoid division by zero
            weights = 1.0 / distances
        else:
            # Binary weights within threshold
            distances = np.sqrt(((coordinates[:, None] - coordinates) ** 2).sum(axis=2))
            weights = (distances <= weight_threshold).astype(float)
            np.fill_diagonal(weights, 0)
        
        # Row-standardize weights
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        weights = weights / row_sums
        
        # Calculate Moran's I
        y_mean = values.mean()
        y_dev = values - y_mean
        
        numerator = n * np.sum(weights * np.outer(y_dev, y_dev))
        denominator = np.sum(weights) * np.sum(y_dev ** 2)
        
        if denominator == 0:
            return 0.0, 1.0
        
        moran_i = numerator / denominator
        
        # Calculate expected value and variance under null hypothesis
        expected_i = -1.0 / (n - 1)
        
        # Simplified p-value calculation (assumes normality)
        # For more accurate p-values, use permutation tests
        b2 = np.sum(y_dev ** 4) / n / (np.sum(y_dev ** 2) / n) ** 2
        variance_i = (n * ((n ** 2 - 3 * n + 3) * np.sum(weights ** 2) - n * np.sum(weights) ** 2 + 3 * np.sum(weights) ** 2) -
                     b2 * ((n ** 2 - n) * np.sum(weights ** 2) - 2 * n * np.sum(weights ** 2) + 6 * np.sum(weights) ** 2)) / \
                    ((n - 1) * (n - 2) * (n - 3) * np.sum(weights) ** 2)
        
        if variance_i <= 0:
            p_value = 1.0
        else:
            z_score = (moran_i - expected_i) / np.sqrt(variance_i)
            # Two-tailed p-value
            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        logger.info(f"Moran's I = {moran_i:.4f}, p-value = {p_value:.4f}")
        
        return moran_i, p_value
    
    @staticmethod
    def local_morans_i(
        values: np.ndarray,
        coordinates: np.ndarray,
        weight_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate Local Moran's I (LISA) for each location.
        
        Args:
            values: Data values at each location
            coordinates: Spatial coordinates (n, 2)
            weight_threshold: Distance threshold for weights
            
        Returns:
            Array of local Moran's I values
        """
        n = len(values)
        
        # Create spatial weights
        if weight_threshold is None:
            distances = np.sqrt(((coordinates[:, None] - coordinates) ** 2).sum(axis=2))
            np.fill_diagonal(distances, np.inf)
            weights = 1.0 / distances
        else:
            distances = np.sqrt(((coordinates[:, None] - coordinates) ** 2).sum(axis=2))
            weights = (distances <= weight_threshold).astype(float)
            np.fill_diagonal(weights, 0)
        
        # Row-standardize
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        weights = weights / row_sums
        
        # Calculate local Moran's I
        y_mean = values.mean()
        y_dev = values - y_mean
        
        local_i = np.zeros(n)
        for i in range(n):
            local_i[i] = y_dev[i] * np.sum(weights[i] * y_dev)
        
        # Normalize
        variance = np.sum(y_dev ** 2) / n
        if variance > 0:
            local_i = local_i / variance
        
        return local_i
    
    @staticmethod
    def spatial_correlogram(
        values: np.ndarray,
        coordinates: np.ndarray,
        distance_bins: Optional[np.ndarray] = None,
        n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Calculate spatial correlogram (autocorrelation at different distances).
        
        Args:
            values: Data values
            coordinates: Spatial coordinates
            distance_bins: Custom distance bins (auto if None)
            n_bins: Number of bins if auto
            
        Returns:
            Dict with 'distances', 'correlations', and 'counts'
        """
        # Calculate pairwise distances
        distances = np.sqrt(((coordinates[:, None] - coordinates) ** 2).sum(axis=2))
        
        # Create distance bins
        if distance_bins is None:
            max_dist = distances[distances < np.inf].max()
            distance_bins = np.linspace(0, max_dist, n_bins + 1)
        
        bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
        correlations = np.zeros(len(bin_centers))
        counts = np.zeros(len(bin_centers))
        
        # Standardize values
        values_std = (values - values.mean()) / values.std()
        
        # Calculate correlation for each distance bin
        for i in range(len(bin_centers)):
            mask = (distances >= distance_bins[i]) & (distances < distance_bins[i + 1])
            
            if mask.sum() > 0:
                # Get pairs in this distance bin
                pairs_i, pairs_j = np.where(mask)
                
                # Calculate correlation
                correlation = np.mean(values_std[pairs_i] * values_std[pairs_j])
                correlations[i] = correlation
                counts[i] = len(pairs_i)
        
        return {
            'distances': bin_centers,
            'correlations': correlations,
            'counts': counts
        }
    
    @staticmethod
    def test_spatial_independence(
        train_coords: np.ndarray,
        test_coords: np.ndarray,
        min_distance: float = 0.0
    ) -> Tuple[bool, float]:
        """
        Test if train and test sets are spatially independent.
        
        Args:
            train_coords: Training coordinates
            test_coords: Test coordinates
            min_distance: Minimum required distance
            
        Returns:
            Tuple of (is_independent, min_observed_distance)
        """
        # Calculate minimum distance between train and test
        min_distances = []
        
        for test_coord in test_coords:
            distances = np.sqrt(np.sum((train_coords - test_coord) ** 2, axis=1))
            min_distances.append(distances.min())
        
        min_observed = np.min(min_distances)
        is_independent = min_observed >= min_distance
        
        if not is_independent:
            logger.warning(f"Train/test sets not spatially independent: "
                         f"min distance = {min_observed:.3f} < {min_distance}")
        
        return is_independent, min_observed