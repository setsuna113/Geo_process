"""Moran Eigenvector Spatial Filtering (MESF) implementation.

Based on regression_analysis_decisions.md lines 69-192.
Includes both standard MESF for small datasets and Nystrom approximation for large datasets.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Union
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class StandardMESF:
    """Standard MESF implementation for datasets < 100k points.
    
    Based on regression_analysis_decisions.md lines 164-192.
    """
    
    def __init__(self, 
                 variance_threshold: float = 0.75,
                 max_eigenvectors: int = 10,
                 min_eigenvectors: int = 5,
                 distance_threshold: float = 100):
        """Initialize Standard MESF.
        
        Args:
            variance_threshold: Cumulative variance to explain (default 0.75 = 75%)
            max_eigenvectors: Maximum number of eigenvectors to select
            min_eigenvectors: Minimum number of eigenvectors to select
            distance_threshold: Distance threshold in km for spatial neighbors
        """
        self.variance_threshold = variance_threshold
        self.max_eigenvectors = max_eigenvectors
        self.min_eigenvectors = min_eigenvectors
        self.distance_threshold = distance_threshold
        
        logger.info(f"Initialized StandardMESF with variance_threshold={variance_threshold}, "
                   f"max_eigenvectors={max_eigenvectors}")
    
    def fit_transform(self, coords: np.ndarray) -> np.ndarray:
        """Compute spatial eigenvectors.
        
        Args:
            coords: Array of shape (n_samples, 2) with [longitude, latitude]
            
        Returns:
            Array of shape (n_samples, n_eigenvectors) with selected eigenvectors
        """
        n_samples = coords.shape[0]
        logger.info(f"Computing MESF for {n_samples:,} samples")
        
        # Create spatial weights matrix
        logger.info("Creating spatial weights matrix...")
        W = self._create_spatial_weights(coords)
        
        # Compute eigenvectors
        logger.info("Computing eigenvectors...")
        eigenvalues, eigenvectors = self._compute_eigenvectors(W)
        
        # Select eigenvectors based on variance
        selected_idx = self._select_eigenvectors(eigenvalues)
        selected_eigenvectors = eigenvectors[:, selected_idx]
        
        n_selected = len(selected_idx)
        cumvar = np.sum(eigenvalues[selected_idx]) / np.sum(np.abs(eigenvalues)) * 100
        
        logger.info(f"Selected {n_selected} eigenvectors explaining {cumvar:.1f}% of spatial variance")
        
        return selected_eigenvectors
    
    def _create_spatial_weights(self, coords: np.ndarray) -> np.ndarray:
        """Create spatial weights matrix using distance threshold.
        
        Uses great circle distance for lat/lon coordinates.
        """
        from geopy.distance import great_circle
        
        n = coords.shape[0]
        W = np.zeros((n, n))
        
        # Compute pairwise distances
        for i in range(n):
            for j in range(i+1, n):
                # Calculate great circle distance
                dist = great_circle(
                    (coords[i, 1], coords[i, 0]),  # (lat, lon)
                    (coords[j, 1], coords[j, 0])
                ).km
                
                # Set weight if within threshold
                if dist <= self.distance_threshold:
                    W[i, j] = 1
                    W[j, i] = 1
        
        # Row-standardize weights
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        W = W / row_sums[:, np.newaxis]
        
        n_neighbors = np.mean(row_sums[row_sums > 0])
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
        """Select eigenvectors based on variance threshold.
        
        Implements variance-based selection from regression_analysis_decisions.md lines 106-117.
        """
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
        
        # Enforce min/max constraints
        n_select = max(self.min_eigenvectors, 
                      min(n_select, self.max_eigenvectors, len(positive_idx)))
        
        selected_idx = positive_idx[:n_select]
        
        logger.info(f"Eigenvalue selection: {n_select} eigenvectors explain "
                   f"{cumvar[n_select-1]*100:.1f}% of positive spatial variance")
        
        return selected_idx


class NystromMESF:
    """Nystrom approximation for MESF on large datasets.
    
    Based on regression_analysis_decisions.md lines 77-158.
    For datasets with >100k points.
    """
    
    def __init__(self,
                 n_landmark: int = 5000,
                 n_sample: int = 25000,
                 variance_threshold: float = 0.75,
                 max_eigenvectors: int = 10,
                 min_eigenvectors: int = 5,
                 distance_threshold: float = 100):
        """Initialize Nystrom MESF.
        
        Args:
            n_landmark: Number of landmark points for eigendecomposition
            n_sample: Total number of sample points
            variance_threshold: Cumulative variance to explain
            max_eigenvectors: Maximum number of eigenvectors
            min_eigenvectors: Minimum number of eigenvectors
            distance_threshold: Distance threshold in km
        """
        self.n_landmark = n_landmark
        self.n_sample = n_sample
        self.variance_threshold = variance_threshold
        self.max_eigenvectors = max_eigenvectors
        self.min_eigenvectors = min_eigenvectors
        self.distance_threshold = distance_threshold
        
        logger.info(f"Initialized NystromMESF with {n_landmark:,} landmarks, "
                   f"{n_sample:,} samples")
    
    def fit_transform(self, coords: np.ndarray) -> np.ndarray:
        """Compute spatial eigenvectors using Nystrom approximation.
        
        Args:
            coords: Array of shape (n_samples, 2) with [longitude, latitude]
            
        Returns:
            Array of shape (n_samples, n_eigenvectors) with selected eigenvectors
        """
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
        
        # Step 2: Compute spatial weights for landmarks only
        logger.info("Computing spatial weights for landmarks...")
        coords_landmark = coords[landmark_idx]
        W_landmark = self._create_landmark_weights(coords_landmark)
        
        # Step 3: Eigendecomposition on small landmark matrix
        logger.info("Computing eigendecomposition on landmark matrix...")
        eigenvalues, eigenvectors_landmark = np.linalg.eigh(W_landmark)
        
        # Step 4: Select eigenvectors based on variance
        selected_idx = self._select_eigenvectors(eigenvalues)
        n_select = len(selected_idx)
        
        # Step 5: Nystrom extension to sample points
        logger.info(f"Extending {n_select} eigenvectors to sample points...")
        eigenvectors_sample = self._nystrom_extension(
            coords[sample_idx], 
            coords_landmark,
            eigenvectors_landmark[:, selected_idx]
        )
        
        # Step 6: Interpolate to full dataset
        logger.info(f"Interpolating to full dataset ({n_total:,} points)...")
        ME_full = self._interpolate_to_full(
            coords, 
            coords[sample_idx],
            eigenvectors_sample
        )
        
        cumvar = np.sum(eigenvalues[selected_idx]) / np.sum(np.abs(eigenvalues)) * 100
        logger.info(f"Selected {n_select} eigenvectors explaining {cumvar:.1f}% of spatial variance")
        
        return ME_full
    
    def _create_landmark_weights(self, coords: np.ndarray) -> np.ndarray:
        """Create spatial weights for landmark points."""
        n = coords.shape[0]
        
        # Use vectorized distance computation for efficiency
        from scipy.spatial.distance import pdist, squareform
        
        # Convert lat/lon to radians for haversine distance
        coords_rad = np.radians(coords)
        
        # Compute pairwise distances using haversine formula
        # This is faster than great_circle for many points
        distances = self._haversine_distances(coords_rad)
        
        # Create binary weights based on distance threshold
        W = (distances <= self.distance_threshold).astype(float)
        np.fill_diagonal(W, 0)  # No self-connections
        
        # Row-standardize
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1
        W = W / row_sums[:, np.newaxis]
        
        return W
    
    def _haversine_distances(self, coords_rad: np.ndarray) -> np.ndarray:
        """Compute haversine distances between all pairs of points.
        
        Args:
            coords_rad: Coordinates in radians [lon, lat]
            
        Returns:
            Distance matrix in km
        """
        n = coords_rad.shape[0]
        distances = np.zeros((n, n))
        
        for i in range(n):
            # Vectorized computation for all j
            dlon = coords_rad[:, 0] - coords_rad[i, 0]
            dlat = coords_rad[:, 1] - coords_rad[i, 1]
            
            a = np.sin(dlat/2)**2 + np.cos(coords_rad[i, 1]) * np.cos(coords_rad[:, 1]) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distances[i, :] = 6371 * c  # Earth radius in km
        
        return distances
    
    def _nystrom_extension(self, 
                          coords_sample: np.ndarray,
                          coords_landmark: np.ndarray,
                          eigenvectors_landmark: np.ndarray) -> np.ndarray:
        """Extend eigenvectors from landmarks to sample points.
        
        Implements Nystrom extension from regression_analysis_decisions.md lines 120-129.
        """
        # Compute distances between samples and landmarks
        coords_sample_rad = np.radians(coords_sample)
        coords_landmark_rad = np.radians(coords_landmark)
        
        n_sample = coords_sample.shape[0]
        n_landmark = coords_landmark.shape[0]
        
        # Compute sample-to-landmark distances
        D_SL = np.zeros((n_sample, n_landmark))
        for i in range(n_sample):
            dlon = coords_landmark_rad[:, 0] - coords_sample_rad[i, 0]
            dlat = coords_landmark_rad[:, 1] - coords_sample_rad[i, 1]
            a = np.sin(dlat/2)**2 + np.cos(coords_sample_rad[i, 1]) * np.cos(coords_landmark_rad[:, 1]) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            D_SL[i, :] = 6371 * c
        
        # Gaussian kernel with 100km bandwidth (from line 123)
        W_SL = np.exp(-D_SL / 100)
        
        # Normalize weights
        W_SL = W_SL / W_SL.sum(axis=1, keepdims=True)
        
        # Project eigenvectors to sample points
        eigenvectors_sample = W_SL @ eigenvectors_landmark
        
        return eigenvectors_sample
    
    def _interpolate_to_full(self,
                            coords_full: np.ndarray,
                            coords_sample: np.ndarray,
                            eigenvectors_sample: np.ndarray) -> np.ndarray:
        """Interpolate eigenvectors to full dataset using k-NN.
        
        Implements interpolation from regression_analysis_decisions.md lines 131-147.
        """
        n_total = coords_full.shape[0]
        n_eigenvectors = eigenvectors_sample.shape[1]
        
        # Use k-nearest neighbors for interpolation
        k = min(50, coords_sample.shape[0])
        nbrs = NearestNeighbors(n_neighbors=k, metric='haversine')
        
        # Convert to radians for haversine metric
        coords_sample_rad = np.radians(coords_sample[:, [1, 0]])  # [lat, lon] for sklearn
        coords_full_rad = np.radians(coords_full[:, [1, 0]])
        
        nbrs.fit(coords_sample_rad)
        distances, indices = nbrs.kneighbors(coords_full_rad)
        
        # Distance-weighted average
        ME_full = np.zeros((n_total, n_eigenvectors))
        
        for i in range(n_eigenvectors):
            # Compute weights (inverse distance)
            weights = 1 / (distances + 0.01)  # Add small value to avoid division by zero
            weights = weights / weights.sum(axis=1, keepdims=True)
            
            # Weighted average of eigenvector values
            ME_full[:, i] = np.sum(weights * eigenvectors_sample[indices, i], axis=1)
        
        return ME_full
    
    def _select_eigenvectors(self, eigenvalues: np.ndarray) -> np.ndarray:
        """Select eigenvectors based on variance threshold."""
        # Same implementation as StandardMESF
        positive_idx = np.where(eigenvalues > 0)[0]
        positive_values = eigenvalues[positive_idx]
        
        if len(positive_values) == 0:
            return np.array([], dtype=int)
        
        cumvar = np.cumsum(positive_values) / np.sum(positive_values)
        n_select = np.searchsorted(cumvar, self.variance_threshold) + 1
        n_select = max(self.min_eigenvectors, 
                      min(n_select, self.max_eigenvectors, len(positive_idx)))
        
        return positive_idx[:n_select]


def compute_mesf(coords: np.ndarray, 
                 config: dict,
                 use_nystrom: Optional[bool] = None) -> pd.DataFrame:
    """Compute MESF eigenvectors with automatic method selection.
    
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
        logger.info(f"Using Nystrom approximation for {n_samples:,} samples")
        mesf = NystromMESF(
            n_landmark=config['nystrom']['n_landmark'],
            n_sample=config['nystrom']['n_sample'],
            variance_threshold=config['mesf']['variance_threshold'],
            max_eigenvectors=config['mesf']['max_eigenvectors'],
            min_eigenvectors=config['mesf']['min_eigenvectors'],
            distance_threshold=config['mesf']['distance_threshold']
        )
    else:
        logger.info(f"Using standard MESF for {n_samples:,} samples")
        mesf = StandardMESF(
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
    df = pd.DataFrame(eigenvectors, columns=columns)
    
    logger.info(f"Computed {n_eigenvectors} spatial eigenvectors")
    
    return df