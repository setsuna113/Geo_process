"""Simplified k-means implementation for biodiversity data."""

import numpy as np
import logging
from typing import Optional
from sklearn.cluster import KMeans
from .kmeans_config import KMeansConfig

logger = logging.getLogger(__name__)


class BiodiversityKMeans:
    """Simplified k-means for biodiversity data.
    
    Uses standard sklearn KMeans with preprocessing.
    """
    
    def __init__(self, config: KMeansConfig):
        """Initialize biodiversity k-means.
        
        Args:
            config: K-means configuration
        """
        self.config = config
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        
    def fit(self, X: np.ndarray, coordinates: Optional[np.ndarray] = None) -> 'BiodiversityKMeans':
        """Fit k-means to biodiversity data.
        
        Args:
            X: Feature data (n_samples, n_features)
            coordinates: Geographic coordinates (optional)
            
        Returns:
            Self for method chaining
        """
        n_samples, n_features = X.shape
        logger.info(f"Fitting k-means on {n_samples} samples with {n_features} features")
        
        # Apply transformations
        X_transformed = self._transform_data(X)
        
        # Apply normalization
        X_normalized = self._normalize_data(X_transformed)
        
        # Run sklearn KMeans
        kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            init=self.config.init,
            n_init=self.config.n_init,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            random_state=self.config.random_state
        )
        
        kmeans.fit(X_normalized)
        
        # Store results
        self.labels_ = kmeans.labels_
        self.cluster_centers_ = kmeans.cluster_centers_
        self.inertia_ = kmeans.inertia_
        self.n_iter_ = kmeans.n_iter_
        
        logger.info(f"K-means converged in {self.n_iter_} iterations")
        
        return self
    
    def _transform_data(self, X: np.ndarray) -> np.ndarray:
        """Apply data transformation."""
        if self.config.transform == 'log1p':
            # Handle negative values
            X_min = X.min()
            if X_min < 0:
                X = X - X_min  # Shift to positive
            return np.log1p(X)
        elif self.config.transform == 'sqrt':
            # Handle negative values
            X_min = X.min()
            if X_min < 0:
                X = X - X_min  # Shift to positive
            return np.sqrt(X)
        else:
            return X
    
    def _normalize_data(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        if self.config.normalize == 'standardize':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        elif self.config.normalize == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            return scaler.fit_transform(X)
        else:
            return X
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data.
        
        Args:
            X: Feature data
            
        Returns:
            Cluster labels
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet")
        
        # Apply same transformations
        X_transformed = self._transform_data(X)
        X_normalized = self._normalize_data(X_transformed)
        
        # Calculate distances to centers
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(X_normalized, self.cluster_centers_)
        
        # Assign to nearest cluster
        return np.argmin(distances, axis=1)