"""Feature weight calculation for biodiversity k-means."""

import numpy as np
import logging
from typing import Optional, Tuple
from .kmeans_config import KMeansConfig

logger = logging.getLogger(__name__)


class FeatureWeightCalculator:
    """Calculate feature weights for biodiversity data."""
    
    def __init__(self, config: KMeansConfig):
        """Initialize weight calculator.
        
        Args:
            config: K-means configuration
        """
        self.config = config
    
    def calculate_weights(self, data: np.ndarray, 
                         coordinates: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate feature weights based on configuration.
        
        Args:
            data: Feature data (n_samples, n_features)
            coordinates: Geographic coordinates (n_samples, 2) [lat, lon]
            
        Returns:
            weights: Feature weights normalized to sum to 1
        """
        if self.config.weight_method == 'fixed':
            weights = np.array(self.config.fixed_weights[:data.shape[1]])
            return weights / weights.sum()
        
        elif self.config.weight_method == 'completeness':
            return self._calculate_completeness_weights(data)
        
        elif self.config.weight_method == 'variance':
            return self._calculate_variance_weights(data)
        
        elif self.config.weight_method == 'auto':
            return self._calculate_auto_weights(data)
        
        else:
            # Equal weights as fallback
            logger.warning(f"Unknown weight method: {self.config.weight_method}. Using equal weights.")
            return np.ones(data.shape[1]) / data.shape[1]
    
    def _calculate_completeness_weights(self, data: np.ndarray) -> np.ndarray:
        """Weight features by data completeness.
        
        Features with more non-missing values get higher weights.
        """
        n_samples, n_features = data.shape
        
        # Calculate completeness per feature
        completeness = 1 - (np.isnan(data).sum(axis=0) / n_samples)
        
        # Avoid zero weights
        completeness = np.maximum(completeness, 0.01)
        
        # Normalize
        weights = completeness / completeness.sum()
        
        logger.info(f"Completeness weights: {weights}")
        logger.info(f"Feature completeness: {completeness}")
        
        return weights
    
    def _calculate_variance_weights(self, data: np.ndarray) -> np.ndarray:
        """Weight features by variance (information content).
        
        Features with higher variance are considered more informative.
        """
        # Calculate variance ignoring NaN
        variances = np.nanvar(data, axis=0)
        
        # Avoid zero weights
        variances = np.maximum(variances, 1e-10)
        
        # Log transform to reduce extreme differences
        log_variances = np.log1p(variances)
        
        # Normalize
        weights = log_variances / log_variances.sum()
        
        logger.info(f"Variance weights: {weights}")
        logger.info(f"Feature variances: {variances}")
        
        return weights
    
    def _calculate_auto_weights(self, data: np.ndarray) -> np.ndarray:
        """Automatically determine best weight calculation method.
        
        Uses completeness for very sparse data, variance otherwise.
        """
        # Calculate overall data completeness
        total_completeness = 1 - (np.isnan(data).sum() / data.size)
        
        logger.info(f"Overall data completeness: {total_completeness:.2%}")
        
        if total_completeness < 0.2:  # Less than 20% complete
            logger.info("Using completeness-based weights due to sparse data")
            return self._calculate_completeness_weights(data)
        else:
            logger.info("Using variance-based weights")
            return self._calculate_variance_weights(data)
    
    def calculate_spatial_weights(self, data: np.ndarray, 
                                 coordinates: np.ndarray,
                                 window_size_km: float = 200.0) -> np.ndarray:
        """Calculate spatially-aware weights.
        
        This creates location-specific weights based on local data patterns.
        
        Args:
            data: Feature data
            coordinates: Geographic coordinates [lat, lon]
            window_size_km: Size of spatial window
            
        Returns:
            weights: Spatially-varying weights (n_samples, n_features)
        """
        n_samples, n_features = data.shape
        spatial_weights = np.zeros((n_samples, n_features))
        
        # Import spatial distance calculation
        from ..som.spatial_utils import haversine_distance
        
        for i in range(n_samples):
            # Find neighbors within window
            distances = haversine_distance(coordinates[i], coordinates)
            neighbors = distances < window_size_km
            
            if neighbors.sum() > 5:  # Need enough neighbors
                # Calculate local weights
                local_data = data[neighbors]
                
                # Local completeness
                local_completeness = 1 - (np.isnan(local_data).sum(axis=0) / len(local_data))
                
                # Local variance
                local_variance = np.nanvar(local_data, axis=0)
                local_variance = np.maximum(local_variance, 1e-10)
                
                # Combine completeness and variance
                combined = local_completeness * np.log1p(local_variance)
                
                # Normalize
                if combined.sum() > 0:
                    spatial_weights[i] = combined / combined.sum()
                else:
                    spatial_weights[i] = np.ones(n_features) / n_features
            else:
                # Use global weights for isolated points
                spatial_weights[i] = self.calculate_weights(data)
        
        return spatial_weights