"""Core k-means implementation for biodiversity data."""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_chunked
import warnings

from .kmeans_config import KMeansConfig
from .adaptive_distance import AdaptivePartialDistance
from .sparse_optimizer import SparseDataOptimizer
from .weight_calculator import FeatureWeightCalculator

logger = logging.getLogger(__name__)


class BiodiversityKMeans:
    """Optimized k-means for biodiversity data with missing values.
    
    This implementation:
    - Handles up to 90% missing data using partial distances
    - Uses adaptive minimum feature thresholds based on geography
    - Implements hierarchical clustering by data quality
    - Optimizes for sparse data with prefiltering
    """
    
    def __init__(self, config: KMeansConfig):
        """Initialize biodiversity k-means.
        
        Args:
            config: K-means configuration
        """
        self.config = config
        self.optimizer = SparseDataOptimizer(config)
        self.weight_calculator = FeatureWeightCalculator(config)
        
        # These will be set during fit
        self.distance_calculator = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.feature_weights_ = None
        
    def fit(self, X: np.ndarray, coordinates: Optional[np.ndarray] = None) -> 'BiodiversityKMeans':
        """Fit k-means to biodiversity data.
        
        Args:
            X: Feature data (n_samples, n_features)
            coordinates: Geographic coordinates (n_samples, 2) [lat, lon]
            
        Returns:
            self: Fitted estimator
        """
        # Comprehensive input validation
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy array, got {type(X)}")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        n_samples, n_features = X.shape
        
        if n_samples == 0:
            raise ValueError("X contains no samples")
        
        if n_features == 0:
            raise ValueError("X contains no features")
        
        # Validate n_clusters
        if not isinstance(self.config.n_clusters, int) or self.config.n_clusters <= 0:
            raise ValueError(f"n_clusters must be a positive integer, got {self.config.n_clusters}")
        
        if n_samples < self.config.n_clusters:
            raise ValueError(f"Number of samples ({n_samples}) must be >= n_clusters ({self.config.n_clusters})")
        
        # Validate other parameters
        if self.config.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.config.max_iter}")
        
        if self.config.n_init <= 0:
            raise ValueError(f"n_init must be positive, got {self.config.n_init}")
        
        if self.config.tol < 0:
            raise ValueError(f"tol must be non-negative, got {self.config.tol}")
        
        # Validate coordinates if provided
        if coordinates is not None:
            if not isinstance(coordinates, np.ndarray):
                raise TypeError(f"coordinates must be a numpy array, got {type(coordinates)}")
            
            if coordinates.shape != (n_samples, 2):
                raise ValueError(f"coordinates must have shape ({n_samples}, 2), got {coordinates.shape}")
            
            # Check for valid latitude/longitude ranges
            if np.any(np.abs(coordinates[:, 0]) > 90):
                raise ValueError("Latitude values must be between -90 and 90")
            
            if np.any(np.abs(coordinates[:, 1]) > 180):
                raise ValueError("Longitude values must be between -180 and 180")
        
        # Calculate feature weights
        self.feature_weights_ = self.weight_calculator.calculate_weights(X, coordinates)
        logger.info(f"Feature weights: {self.feature_weights_}")
        
        # Initialize distance calculator with weights
        self.distance_calculator = AdaptivePartialDistance(self.config, self.feature_weights_)
        
        # Prefilter if configured
        if self.config.prefilter_empty:
            X_filtered, valid_mask = self.optimizer.prefilter_data(
                X, self.config.min_features_prefilter
            )
            coords_filtered = coordinates[valid_mask] if coordinates is not None else None
            
            # Fit on filtered data
            self._fit_hierarchical(X_filtered, coords_filtered)
            
            # Map labels back to original indices
            self.labels_ = np.full(len(X), -1, dtype=int)
            self.labels_[valid_mask] = self._labels_filtered
            
        else:
            # Fit on all data
            self._fit_hierarchical(X, coordinates)
        
        return self
    
    def _fit_hierarchical(self, X: np.ndarray, coordinates: Optional[np.ndarray] = None):
        """Hierarchical fitting based on data quality."""
        
        # Stratify by data quality
        strata = self.optimizer.stratify_by_quality(X)
        
        # Validate strata indices are within bounds
        n_samples = len(X)
        for quality, indices in strata.items():
            if len(indices) > 0:
                if np.any(indices < 0) or np.any(indices >= n_samples):
                    raise ValueError(f"Invalid indices in {quality} quality stratum: out of bounds [0, {n_samples})")
        
        if len(strata['high']) >= self.config.n_clusters:
            # Fit initial centers on high-quality data
            logger.info("Fitting initial centers on high-quality data")
            self._fit_sklearn_kmeans(X[strata['high']], coordinates[strata['high']] if coordinates is not None else None)
            
            # Create labels array
            labels = np.full(len(X), -1, dtype=int)
            labels[strata['high']] = self._high_quality_labels
            
            # Assign medium quality data
            if len(strata['medium']) > 0:
                logger.info("Assigning medium-quality data")
                medium_labels = self._assign_to_clusters(
                    X[strata['medium']], 
                    coordinates[strata['medium']] if coordinates is not None else None
                )
                labels[strata['medium']] = medium_labels
            
            # Assign low quality data
            if len(strata['low']) > 0:
                logger.info("Assigning low-quality data")
                low_labels = self._assign_to_clusters(
                    X[strata['low']], 
                    coordinates[strata['low']] if coordinates is not None else None
                )
                labels[strata['low']] = low_labels
            
            self._labels_filtered = labels
            
        else:
            # Not enough high-quality data, use custom implementation
            logger.warning("Not enough high-quality data for hierarchical fitting, using full dataset")
            self._fit_custom_kmeans(X, coordinates)
    
    def _fit_sklearn_kmeans(self, X: np.ndarray, coordinates: Optional[np.ndarray] = None):
        """Fit using sklearn KMeans on high-quality data."""
        
        # For high-quality data, we can use sklearn's efficient implementation
        # with a custom metric
        
        if self.config.distance_metric == 'euclidean':
            # Can use sklearn directly
            km = KMeans(
                n_clusters=self.config.n_clusters,
                init=self.config.init,
                n_init=self.config.n_init,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
            km.fit(X)
            
            self.cluster_centers_ = km.cluster_centers_
            self._high_quality_labels = km.labels_
            self.inertia_ = km.inertia_
            self.n_iter_ = km.n_iter_
            
        else:
            # Need custom implementation for Bray-Curtis
            self._fit_custom_kmeans(X, coordinates)
            self._high_quality_labels = self._labels_filtered
    
    def _fit_custom_kmeans(self, X: np.ndarray, coordinates: Optional[np.ndarray] = None):
        """Custom k-means implementation for partial distances."""
        
        n_samples, n_features = X.shape
        
        # Initialize centers
        centers = self._initialize_centers(X, coordinates)
        
        best_labels = None
        best_inertia = np.inf
        
        # Multiple initializations
        for init_idx in range(self.config.n_init):
            if init_idx > 0:
                centers = self._initialize_centers(X, coordinates)
            
            labels = np.zeros(n_samples, dtype=int)
            prev_labels = np.ones(n_samples, dtype=int)
            
            # Main k-means loop
            for iteration in range(self.config.max_iter):
                # Assignment step
                for i in range(n_samples):
                    coord_i = coordinates[i] if coordinates is not None else None
                    min_dist = np.inf
                    best_cluster = 0
                    
                    for k in range(self.config.n_clusters):
                        dist = self.distance_calculator.calculate_distance(
                            X[i], centers[k], coord_i, None
                        )
                        if dist < min_dist:
                            min_dist = dist
                            best_cluster = k
                    
                    labels[i] = best_cluster
                
                # Check convergence
                if np.array_equal(labels, prev_labels):
                    logger.debug(f"Converged at iteration {iteration}")
                    break
                
                # Use in-place copy for better performance
                np.copyto(prev_labels, labels)
                
                # Update step
                for k in range(self.config.n_clusters):
                    cluster_mask = labels == k
                    if cluster_mask.sum() > 0:
                        # Update center as mean of assigned points
                        centers[k] = np.nanmean(X[cluster_mask], axis=0)
            
            # Calculate inertia
            inertia = self._calculate_inertia(X, labels, centers, coordinates)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                self.cluster_centers_ = centers.copy()
        
        self._labels_filtered = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = iteration + 1
    
    def _initialize_centers(self, X: np.ndarray, coordinates: Optional[np.ndarray] = None) -> np.ndarray:
        """Initialize cluster centers."""
        
        n_samples, n_features = X.shape
        
        if self.config.init == 'k-means++':
            # K-means++ initialization adapted for partial distances
            centers = np.empty((self.config.n_clusters, n_features))
            
            # Choose first center randomly from data points
            center_idx = np.random.choice(n_samples)
            if center_idx < 0 or center_idx >= n_samples:
                raise ValueError(f"Invalid center index: {center_idx}")
            centers[0] = X[center_idx]  # Direct assignment, no copy needed
            
            # Choose remaining centers
            for k in range(1, self.config.n_clusters):
                # Calculate distances to nearest center - vectorized for performance
                min_distances = np.full(n_samples, np.inf)
                
                # Pre-compute coordinates for efficiency
                coords = coordinates if coordinates is not None else [None] * n_samples
                
                for i in range(n_samples):
                    coord_i = coords[i] if coords is not None else None
                    # Only calculate to previous centers
                    for j in range(k):
                        dist = self.distance_calculator.calculate_distance(
                            X[i], centers[j], coord_i, None
                        )
                        if dist < min_distances[i]:
                            min_distances[i] = dist
                
                # Choose next center with probability proportional to squared distance
                # Handle infinite distances
                finite_mask = np.isfinite(min_distances)
                if finite_mask.sum() > 0:
                    probs = min_distances.copy()
                    probs[~finite_mask] = 0
                    probs = probs ** 2
                    # Protect against division by zero
                    prob_sum = probs.sum()
                    if prob_sum > 0:
                        probs = probs / prob_sum
                        center_idx = np.random.choice(n_samples, p=probs)
                    else:
                        # All distances are 0 or inf - random selection
                        center_idx = np.random.choice(n_samples)
                else:
                    # Fallback to random selection
                    center_idx = np.random.choice(n_samples)
                
                centers[k] = X[center_idx]  # Direct assignment
            
        else:
            # Random initialization - direct indexing is more efficient
            indices = np.random.choice(n_samples, self.config.n_clusters, replace=False)
            centers = X[indices]
        
        return centers
    
    def _assign_to_clusters(self, X: np.ndarray, coordinates: Optional[np.ndarray] = None) -> np.ndarray:
        """Assign samples to nearest cluster centers."""
        
        n_samples = len(X)
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            coord_i = coordinates[i] if coordinates is not None else None
            min_dist = np.inf
            best_cluster = 0
            
            for k in range(self.config.n_clusters):
                dist = self.distance_calculator.calculate_distance(
                    X[i], self.cluster_centers_[k], coord_i, None
                )
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = k
            
            labels[i] = best_cluster
        
        return labels
    
    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray, 
                          centers: np.ndarray, coordinates: Optional[np.ndarray] = None) -> float:
        """Calculate sum of squared distances to nearest centers."""
        
        inertia = 0.0
        for i in range(len(X)):
            coord_i = coordinates[i] if coordinates is not None else None
            dist = self.distance_calculator.calculate_distance(
                X[i], centers[labels[i]], coord_i, None
            )
            if np.isfinite(dist):
                inertia += dist ** 2
        
        return inertia
    
    def predict(self, X: np.ndarray, coordinates: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict cluster labels for new data.
        
        Args:
            X: Feature data
            coordinates: Geographic coordinates
            
        Returns:
            labels: Cluster assignments
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self._assign_to_clusters(X, coordinates)
    
    def fit_predict(self, X: np.ndarray, coordinates: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit model and return cluster labels.
        
        Args:
            X: Feature data
            coordinates: Geographic coordinates
            
        Returns:
            labels: Cluster assignments
        """
        self.fit(X, coordinates)
        return self.labels_