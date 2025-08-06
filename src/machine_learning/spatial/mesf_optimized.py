"""Optimized MESF implementation with vectorized distance calculations.

This module provides performance improvements over the original MESF implementation
by using vectorized distance calculations and spatial indexing.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import haversine_distances

logger = logging.getLogger(__name__)


class StandardMESFOptimized:
    """Optimized Standard MESF with vectorized distance calculations.
    
    Performance improvements:
    - Uses sklearn's haversine_distances (vectorized)
    - Uses scipy's cdist for euclidean distances when appropriate
    - Eliminates nested loops in distance calculations
    """
    
    def __init__(self,
                 variance_threshold: float = 0.75,
                 max_eigenvectors: int = 10,
                 min_eigenvectors: int = 5,
                 distance_threshold: float = 100):
        """Initialize optimized MESF.
        
        Args:
            variance_threshold: Cumulative variance to explain
            max_eigenvectors: Maximum number of eigenvectors to select
            min_eigenvectors: Minimum number of eigenvectors to select
            distance_threshold: Distance threshold in km for spatial neighbors
        """
        self.variance_threshold = variance_threshold
        self.max_eigenvectors = max_eigenvectors
        self.min_eigenvectors = min_eigenvectors
        self.distance_threshold = distance_threshold
        
        logger.info(f"Initialized StandardMESFOptimized with variance_threshold={variance_threshold}, "
                   f"max_eigenvectors={max_eigenvectors}")
    
    def fit_transform(self, coords: np.ndarray) -> np.ndarray:
        """Compute spatial eigenvectors using optimized distance calculations.
        
        Args:
            coords: Array of shape (n_samples, 2) with [longitude, latitude]
            
        Returns:
            Array of shape (n_samples, n_eigenvectors) with selected eigenvectors
        """
        n_samples = coords.shape[0]
        logger.info(f"Computing MESF for {n_samples} samples")
        
        # Create spatial weights matrix using vectorized distance computation
        logger.info("Creating spatial weights matrix...")
        W = self._create_spatial_weights_vectorized(coords)
        
        # Compute eigenvectors
        logger.info("Computing eigenvectors...")
        eigenvalues, eigenvectors = self._compute_eigenvectors(W)
        
        # Select eigenvectors based on variance
        selected_idx = self._select_eigenvectors(eigenvalues)
        selected_eigenvectors = eigenvectors[:, selected_idx]
        
        logger.info(f"StandardMESFOptimized: Selected {len(selected_idx)} eigenvectors")
        
        return selected_eigenvectors
    
    def _create_spatial_weights_vectorized(self, coords: np.ndarray) -> np.ndarray:
        """Create spatial weights matrix using vectorized haversine distance.
        
        This is the main optimization - replaces O(n²) nested loops with
        vectorized computation.
        """
        n = coords.shape[0]
        
        # Convert coordinates to radians for haversine calculation
        coords_rad = np.radians(coords[:, [1, 0]])  # Convert to [lat, lon] in radians
        
        # Compute all pairwise distances at once using sklearn's optimized implementation
        # haversine_distances returns distances in radians, need to convert to km
        distances_rad = haversine_distances(coords_rad)
        distances_km = distances_rad * 6371  # Earth radius in km
        
        # Create binary weights based on distance threshold
        W = (distances_km <= self.distance_threshold).astype(float)
        np.fill_diagonal(W, 0)  # No self-connections
        
        # Row-standardize weights
        row_sums = W.sum(axis=1)
        n_neighbors = np.mean(row_sums[row_sums > 0]) if np.any(row_sums > 0) else 0
        
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        W = W / row_sums[:, np.newaxis]
        
        logger.info(f"Average number of neighbors: {n_neighbors:.1f}")
        
        return W
    
    def _compute_eigenvectors(self, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvectors of spatial weights matrix."""
        # For symmetric matrix, use more efficient eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(W)
        
        # Sort by eigenvalue magnitude (descending)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def _select_eigenvectors(self, eigenvalues: np.ndarray) -> np.ndarray:
        """Select eigenvectors based on variance threshold."""
        # Only consider positive eigenvalues
        positive_idx = np.where(eigenvalues > 0)[0]
        positive_values = eigenvalues[positive_idx]
        
        if len(positive_values) == 0:
            logger.warning("No positive eigenvalues found, returning empty selection")
            return np.array([], dtype=int)
        
        # Calculate cumulative variance
        cumvar = np.cumsum(positive_values) / np.sum(positive_values)
        
        # Find number of eigenvectors needed to explain threshold variance
        n_select = np.searchsorted(cumvar, self.variance_threshold) + 1
        
        # Enforce constraints
        n_select = min(n_select, len(positive_idx))
        n_select = min(n_select, self.max_eigenvectors)
        
        # Apply min constraint only if we have enough eigenvalues
        if len(positive_idx) >= self.min_eigenvectors:
            n_select = max(n_select, self.min_eigenvectors)
        
        # Ensure we don't exceed what's available
        n_select = min(n_select, len(positive_idx))
        
        selected_idx = positive_idx[:n_select]
        
        # Calculate actual variance explained
        if n_select > 0 and len(cumvar) > 0:
            actual_idx = min(n_select - 1, len(cumvar) - 1)
            variance_explained = cumvar[actual_idx] * 100
            logger.info(f"Selected {n_select} eigenvectors explaining "
                       f"{variance_explained:.1f}% of positive spatial variance")
        
        return selected_idx


class NystromMESFOptimized:
    """Optimized Nystrom approximation for MESF on large datasets.
    
    Performance improvements:
    - Vectorized distance calculations for landmark weights
    - Batch processing for Nystrom extension
    - Efficient k-NN interpolation using sklearn's BallTree
    """
    
    def __init__(self,
                 n_landmark: int = 5000,
                 n_sample: int = 25000,
                 variance_threshold: float = 0.75,
                 max_eigenvectors: int = 10,
                 min_eigenvectors: int = 5,
                 distance_threshold: float = 100):
        """Initialize optimized Nystrom MESF."""
        self.n_landmark = n_landmark
        self.n_sample = n_sample
        self.variance_threshold = variance_threshold
        self.max_eigenvectors = max_eigenvectors
        self.min_eigenvectors = min_eigenvectors
        self.distance_threshold = distance_threshold
        
        logger.info(f"Initialized NystromMESFOptimized with {n_landmark:,} landmarks, "
                   f"{n_sample:,} samples")
    
    def fit_transform(self, coords: np.ndarray) -> np.ndarray:
        """Compute spatial eigenvectors using optimized Nystrom approximation."""
        n_total = coords.shape[0]
        logger.info(f"Computing Nystrom MESF for {n_total:,} samples")
        
        # Step 1: Select landmark and sample points
        logger.info(f"Selecting {self.n_landmark:,} landmark points...")
        landmark_idx = np.random.choice(n_total, 
                                       min(self.n_landmark, n_total), 
                                       replace=False)
        sample_idx = np.random.choice(n_total, 
                                     min(self.n_sample, n_total), 
                                     replace=False)
        
        # Step 2: Compute spatial weights for landmarks only (vectorized)
        logger.info("Computing spatial weights for landmarks...")
        coords_landmark = coords[landmark_idx]
        W_landmark = self._create_landmark_weights_vectorized(coords_landmark)
        
        # Step 3: Eigendecomposition on small landmark matrix
        logger.info("Computing eigendecomposition on landmark matrix...")
        eigenvalues, eigenvectors_landmark = np.linalg.eigh(W_landmark)
        
        # Step 4: Select eigenvectors based on variance
        selected_idx = self._select_eigenvectors(eigenvalues)
        n_select = len(selected_idx)
        
        # Step 5: Nystrom extension to sample points (vectorized)
        logger.info(f"Extending {n_select} eigenvectors to sample points...")
        eigenvectors_sample = self._nystrom_extension_vectorized(
            coords[sample_idx], 
            coords_landmark,
            eigenvectors_landmark[:, selected_idx]
        )
        
        # Step 6: Interpolate to full dataset using efficient k-NN
        logger.info(f"Interpolating to full dataset ({n_total:,} points)...")
        ME_full = self._interpolate_to_full_optimized(
            coords, 
            coords[sample_idx],
            eigenvectors_sample
        )
        
        cumvar = np.sum(eigenvalues[selected_idx]) / np.sum(np.abs(eigenvalues)) * 100
        logger.info(f"Selected {n_select} eigenvectors explaining {cumvar:.1f}% of spatial variance")
        
        return ME_full
    
    def _create_landmark_weights_vectorized(self, coords: np.ndarray) -> np.ndarray:
        """Create spatial weights for landmark points using vectorized operations."""
        # Convert coordinates to radians
        coords_rad = np.radians(coords[:, [1, 0]])  # [lat, lon] in radians
        
        # Compute all pairwise distances at once
        distances_rad = haversine_distances(coords_rad)
        distances_km = distances_rad * 6371  # Earth radius in km
        
        # Create binary weights based on distance threshold
        W = (distances_km <= self.distance_threshold).astype(float)
        np.fill_diagonal(W, 0)  # No self-connections
        
        # Row-standardize
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1
        W = W / row_sums[:, np.newaxis]
        
        return W
    
    def _nystrom_extension_vectorized(self, 
                                     coords_sample: np.ndarray,
                                     coords_landmark: np.ndarray,
                                     eigenvectors_landmark: np.ndarray) -> np.ndarray:
        """Extend eigenvectors from landmarks to sample points using vectorized ops."""
        # Convert to radians
        coords_sample_rad = np.radians(coords_sample[:, [1, 0]])
        coords_landmark_rad = np.radians(coords_landmark[:, [1, 0]])
        
        # Compute all sample-to-landmark distances at once
        distances_rad = haversine_distances(coords_sample_rad, coords_landmark_rad)
        D_SL = distances_rad * 6371  # Convert to km
        
        # Gaussian kernel with 100km bandwidth
        W_SL = np.exp(-D_SL / 100)
        
        # Normalize weights
        W_SL = W_SL / W_SL.sum(axis=1, keepdims=True)
        
        # Project eigenvectors to sample points
        eigenvectors_sample = W_SL @ eigenvectors_landmark
        
        return eigenvectors_sample
    
    def _interpolate_to_full_optimized(self,
                                      coords_full: np.ndarray,
                                      coords_sample: np.ndarray,
                                      eigenvectors_sample: np.ndarray) -> np.ndarray:
        """Interpolate eigenvectors to full dataset using efficient k-NN.
        
        Uses sklearn's BallTree for fast nearest neighbor search on sphere.
        """
        from sklearn.neighbors import BallTree
        
        n_full = coords_full.shape[0]
        n_eigenvectors = eigenvectors_sample.shape[1]
        
        # Convert to radians for BallTree with haversine metric
        coords_full_rad = np.radians(coords_full[:, [1, 0]])
        coords_sample_rad = np.radians(coords_sample[:, [1, 0]])
        
        # Build BallTree for efficient k-NN search
        tree = BallTree(coords_sample_rad, metric='haversine')
        
        # Find k nearest neighbors for each point in full dataset
        k = min(50, len(coords_sample))
        distances, indices = tree.query(coords_full_rad, k=k)
        
        # Convert distances to km
        distances = distances * 6371
        
        # Compute weights using Gaussian kernel
        weights = np.exp(-distances / 100)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Interpolate eigenvectors
        ME_full = np.zeros((n_full, n_eigenvectors))
        for i in range(n_full):
            ME_full[i] = weights[i] @ eigenvectors_sample[indices[i]]
        
        return ME_full
    
    def _select_eigenvectors(self, eigenvalues: np.ndarray) -> np.ndarray:
        """Select eigenvectors based on variance threshold."""
        positive_idx = np.where(eigenvalues > 0)[0]
        
        if len(positive_idx) == 0:
            return np.array([], dtype=int)
        
        positive_values = eigenvalues[positive_idx]
        cumvar = np.cumsum(positive_values) / np.sum(positive_values)
        
        n_select = np.searchsorted(cumvar, self.variance_threshold) + 1
        n_select = max(self.min_eigenvectors, 
                      min(n_select, self.max_eigenvectors, len(positive_idx)))
        
        return positive_idx[:n_select]


def compute_mesf_optimized(coords: np.ndarray, 
                          config: dict,
                          use_nystrom: Optional[bool] = None) -> pd.DataFrame:
    """Compute MESF eigenvectors with optimized distance calculations.
    
    This is a drop-in replacement for compute_mesf with better performance.
    
    Args:
        coords: Array of shape (n_samples, 2) with [longitude, latitude]
        config: Spatial configuration dictionary
        use_nystrom: Force use of Nystrom approximation. If None, auto-detect.
        
    Returns:
        DataFrame with columns ME1, ME2, ..., MEn
    """
    n_samples = coords.shape[0]
    
    # Auto-detect method if not specified
    if use_nystrom is None:
        nystrom_config = config.get('nystrom', {})
        threshold = nystrom_config.get('size_threshold', 100000)
        use_nystrom = n_samples > threshold and nystrom_config.get('enable_for_large_data', True)
    
    if use_nystrom:
        logger.info(f"Using optimized Nystrom approximation for {n_samples:,} samples")
        mesf = NystromMESFOptimized(
            n_landmark=config['nystrom']['n_landmark'],
            n_sample=config['nystrom']['n_sample'],
            variance_threshold=config['mesf']['variance_threshold'],
            max_eigenvectors=config['mesf']['max_eigenvectors'],
            min_eigenvectors=config['mesf']['min_eigenvectors'],
            distance_threshold=config['mesf']['distance_threshold']
        )
    else:
        logger.info(f"Using optimized standard MESF for {n_samples:,} samples")
        mesf = StandardMESFOptimized(
            variance_threshold=config['mesf']['variance_threshold'],
            max_eigenvectors=config['mesf']['max_eigenvectors'],
            min_eigenvectors=config['mesf']['min_eigenvectors'],
            distance_threshold=config['mesf']['distance_threshold']
        )
    
    # Compute eigenvectors
    eigenvectors = mesf.fit_transform(coords)
    
    # Convert to DataFrame
    n_eigenvectors = eigenvectors.shape[1]
    columns = [f'ME{i+1}' for i in range(n_eigenvectors)]
    mesf_df = pd.DataFrame(eigenvectors, columns=columns)
    
    logger.info(f"Computed {n_eigenvectors} spatial eigenvectors")
    
    return mesf_df