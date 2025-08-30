from src.abstractions import get_logger
"""Spatial feature engineering for biodiversity clustering.

This module implements advanced spatial preprocessing techniques including:
- Spatial autocorrelation features (Moran's I)
- Spatial lag variables
- Multi-scale spatial statistics
- Moran eigenvector mapping (MEM)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import logging

logger = get_logger(__name__)


class SpatialFeatureEngineering:
    """Generate spatial features for biodiversity clustering."""
    
    def __init__(self, 
                 neighbor_method: str = 'knn',
                 n_neighbors: int = 8,
                 distance_threshold: float = None,
                 scales: List[float] = None,
                 use_pmem: bool = False,
                 pmem_dwf: str = 'exponential'):
        """Initialize spatial feature engineering.
        
        Args:
            neighbor_method: 'knn' or 'distance' for neighbor definition
            n_neighbors: Number of neighbors for knn method
            distance_threshold: Distance threshold for distance method
            scales: List of scales (km) for multi-scale features
            use_pmem: Use predictive MEM instead of standard MEM
            pmem_dwf: Distance-weighting function for pMEM ('exponential', 'gaussian', 'linear')
        """
        self.neighbor_method = neighbor_method
        self.n_neighbors = n_neighbors
        self.distance_threshold = distance_threshold
        self.scales = scales or [5, 10, 25, 50, 100]
        self.use_pmem = use_pmem
        self.pmem_dwf = pmem_dwf
        
    def fit_transform(self, 
                      data: np.ndarray,
                      coordinates: np.ndarray,
                      feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Generate all spatial features.
        
        Args:
            data: Feature matrix (n_samples, n_features)
            coordinates: Lat/lon coordinates (n_samples, 2)
            feature_names: Original feature names
            
        Returns:
            Augmented feature matrix and new feature names
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(data.shape[1])]
            
        # Start with original features
        features = [data]
        new_names = list(feature_names)
        
        # Add spatial lag features
        logger.info("Creating spatial lag features...")
        lag_features, lag_names = self._create_spatial_lags(data, coordinates, feature_names)
        features.append(lag_features)
        new_names.extend(lag_names)
        
        # Add local Moran's I
        logger.info("Calculating local Moran's I...")
        morans_features, morans_names = self._calculate_local_morans(data, coordinates, feature_names)
        features.append(morans_features)
        new_names.extend(morans_names)
        
        # Add multi-scale statistics
        logger.info("Computing multi-scale spatial statistics...")
        multiscale_features, multiscale_names = self._multiscale_statistics(data, coordinates, feature_names)
        features.append(multiscale_features)
        new_names.extend(multiscale_names)
        
        # Add Moran eigenvectors (standard or predictive)
        if self.use_pmem:
            logger.info("Computing predictive Moran eigenvector maps...")
            # Use total species richness as target for pMEM
            target = np.nansum(data, axis=1)
            mem_features, mem_names = self._predictive_moran_eigenvectors(
                coordinates, target=target, n_components=20, dwf_type=self.pmem_dwf
            )
        else:
            logger.info("Computing Moran eigenvector maps...")
            mem_features, mem_names = self._moran_eigenvector_maps(coordinates, n_components=20)
        
        features.append(mem_features)
        new_names.extend(mem_names)
        
        # Combine all features
        combined = np.hstack(features)
        logger.info(f"Created {combined.shape[1]} features from {data.shape[1]} original features")
        
        return combined, new_names
    
    def _create_spatial_lags(self, 
                             data: np.ndarray,
                             coordinates: np.ndarray,
                             feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Create spatial lag variables (neighbor-averaged features)."""
        # Build spatial weights matrix
        W = self._build_spatial_weights(coordinates)
        
        # Calculate spatial lags
        lag_features = W @ data
        
        # Generate names
        lag_names = [f'{name}_spatial_lag' for name in feature_names]
        
        return lag_features, lag_names
    
    def _calculate_local_morans(self,
                                data: np.ndarray,
                                coordinates: np.ndarray,
                                feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Calculate local Moran's I for each feature."""
        W = self._build_spatial_weights(coordinates)
        n = data.shape[0]
        
        morans_features = []
        morans_names = []
        
        for i, name in enumerate(feature_names):
            # Skip if all values are missing
            valid_mask = ~np.isnan(data[:, i])
            if not np.any(valid_mask):
                continue
                
            # Standardize feature
            x = data[:, i].copy()
            x[~valid_mask] = 0  # Set NaN to 0 for calculation
            z = (x - np.mean(x[valid_mask])) / np.std(x[valid_mask])
            
            # Calculate local Moran's I
            lag_z = W @ z
            local_morans = z * lag_z
            
            # Mask invalid values
            local_morans[~valid_mask] = np.nan
            
            morans_features.append(local_morans)
            morans_names.append(f'{name}_local_morans')
            
        if morans_features:
            return np.column_stack(morans_features), morans_names
        else:
            return np.empty((n, 0)), []
    
    def _multiscale_statistics(self,
                               data: np.ndarray,
                               coordinates: np.ndarray,
                               feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Compute statistics at multiple spatial scales."""
        features = []
        names = []
        
        # Convert coordinates to projected system (simple equirectangular)
        coords_km = self._latlon_to_km(coordinates)
        tree = cKDTree(coords_km)
        
        for scale in self.scales:
            for i, feat_name in enumerate(feature_names):
                # Find neighbors within scale
                neighbor_lists = tree.query_ball_tree(tree, r=scale)
                
                # Calculate statistics
                scale_mean = np.zeros(data.shape[0])
                scale_std = np.zeros(data.shape[0])
                scale_range = np.zeros(data.shape[0])
                
                for j, neighbors in enumerate(neighbor_lists):
                    if len(neighbors) > 1:
                        values = data[neighbors, i]
                        valid_values = values[~np.isnan(values)]
                        
                        if len(valid_values) > 0:
                            scale_mean[j] = np.mean(valid_values)
                            scale_std[j] = np.std(valid_values) if len(valid_values) > 1 else 0
                            scale_range[j] = np.ptp(valid_values)
                        else:
                            scale_mean[j] = np.nan
                            scale_std[j] = np.nan
                            scale_range[j] = np.nan
                    else:
                        scale_mean[j] = data[j, i]
                        scale_std[j] = 0
                        scale_range[j] = 0
                
                features.extend([scale_mean, scale_std, scale_range])
                names.extend([
                    f'{feat_name}_mean_{scale}km',
                    f'{feat_name}_std_{scale}km',
                    f'{feat_name}_range_{scale}km'
                ])
        
        return np.column_stack(features), names
    
    def _moran_eigenvector_maps(self,
                                 coordinates: np.ndarray,
                                 n_components: int = 20) -> Tuple[np.ndarray, List[str]]:
        """Generate Moran eigenvector maps (spatial eigenvectors)."""
        # Build spatial weights matrix
        W = self._build_spatial_weights(coordinates)
        
        # Double-center the weights matrix
        n = W.shape[0]
        row_sums = W.sum(axis=1).A1
        total_sum = row_sums.sum()
        
        # Create centered matrix
        WC = W.copy()
        WC = WC - (row_sums[:, np.newaxis] / n)
        WC = WC - (row_sums[np.newaxis, :] / n)
        WC = WC + (total_sum / (n * n))
        
        # Extract eigenvectors with positive eigenvalues
        try:
            eigenvals, eigenvecs = eigsh(WC, k=min(n_components, n-2), which='LA')
            
            # Keep only positive eigenvalues
            positive_mask = eigenvals > 1e-10
            eigenvecs = eigenvecs[:, positive_mask]
            n_kept = eigenvecs.shape[1]
            
            names = [f'MEM_{i+1}' for i in range(n_kept)]
            
            logger.info(f"Extracted {n_kept} Moran eigenvectors")
            return eigenvecs, names
            
        except Exception as e:
            logger.warning(f"Failed to compute Moran eigenvectors: {e}")
            return np.empty((n, 0)), []
    
    def _predictive_moran_eigenvectors(self, 
                                       coordinates: np.ndarray, 
                                       target: np.ndarray = None,
                                       n_components: int = 20,
                                       dwf_type: str = 'exponential') -> Tuple[np.ndarray, List[str]]:
        """Compute Predictive Moran's Eigenvector Maps (pMEM).
        
        Enhanced version of MEM that optimizes eigenvector selection for prediction.
        Based on Guénard & Legendre (2024) methodology.
        
        Args:
            coordinates: Spatial coordinates
            target: Optional target variable to optimize prediction for
            n_components: Number of eigenvectors to extract
            dwf_type: Distance-weighting function ('exponential', 'gaussian', 'linear')
            
        Returns:
            pMEM features and names
        """
        # Build distance-weighted spatial weights
        W = self._build_predictive_weights(coordinates, dwf_type)
        
        # Double-center the weights matrix
        n = W.shape[0]
        row_sums = W.sum(axis=1).A1
        total_sum = row_sums.sum()
        
        WC = W.copy()
        WC = WC - (row_sums[:, np.newaxis] / n)
        WC = WC - (row_sums[np.newaxis, :] / n)
        WC = WC + (total_sum / (n * n))
        
        # Extract eigenvectors
        try:
            eigenvals, eigenvecs = eigsh(WC, k=min(n_components * 2, n-2), which='LA')
            
            # If target provided, select eigenvectors that best predict it
            if target is not None and not np.all(np.isnan(target)):
                # Calculate correlation with target
                valid_mask = ~np.isnan(target)
                correlations = np.abs([
                    np.corrcoef(eigenvecs[valid_mask, i], target[valid_mask])[0, 1]
                    for i in range(eigenvecs.shape[1])
                ])
                
                # Select top n_components by correlation
                top_indices = np.argsort(correlations)[-n_components:][::-1]
                eigenvecs = eigenvecs[:, top_indices]
                eigenvals = eigenvals[top_indices]
            else:
                # Standard selection: positive eigenvalues
                positive_mask = eigenvals > 1e-10
                eigenvecs = eigenvecs[:, positive_mask][:, :n_components]
            
            n_kept = eigenvecs.shape[1]
            names = [f'pMEM_{i+1}' for i in range(n_kept)]
            
            logger.info(f"Extracted {n_kept} predictive Moran eigenvectors")
            return eigenvecs, names
            
        except Exception as e:
            logger.warning(f"Failed to compute pMEM: {e}")
            return np.empty((n, 0)), []
    
    def _build_predictive_weights(self, coordinates: np.ndarray, dwf_type: str) -> csr_matrix:
        """Build predictive spatial weights with distance-weighting function."""
        n = coordinates.shape[0]
        tree = cKDTree(coordinates)
        
        # Get all pairwise distances within threshold
        if self.distance_threshold:
            threshold = self.distance_threshold
        else:
            # Use adaptive threshold based on nearest neighbors
            distances, _ = tree.query(coordinates, k=self.n_neighbors+1)
            threshold = np.median(distances[:, -1]) * 2
        
        # Find all neighbors within threshold
        pairs = tree.query_pairs(threshold, output_type='ndarray')
        
        if len(pairs) == 0:
            logger.warning("No neighbor pairs found within threshold")
            return csr_matrix((n, n))
        
        # Calculate weights based on DWF type
        weights = []
        row_idx = []
        col_idx = []
        
        for i, j in pairs:
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            
            if dwf_type == 'exponential':
                w = np.exp(-dist / threshold)
            elif dwf_type == 'gaussian':
                w = np.exp(-(dist / threshold) ** 2)
            elif dwf_type == 'linear':
                w = max(0, 1 - dist / threshold)
            else:
                w = 1.0  # Binary weights
            
            # Add symmetric entries
            row_idx.extend([i, j])
            col_idx.extend([j, i])
            weights.extend([w, w])
        
        W = csr_matrix((weights, (row_idx, col_idx)), shape=(n, n))
        
        # Row-standardize
        row_sums = W.sum(axis=1).A1
        row_sums[row_sums == 0] = 1
        W = W.multiply(1 / row_sums[:, np.newaxis])
        
        return W
    
    def _build_spatial_weights(self, coordinates: np.ndarray) -> csr_matrix:
        """Build spatial weights matrix."""
        n = coordinates.shape[0]
        
        if self.neighbor_method == 'knn':
            # KNN-based weights
            tree = cKDTree(coordinates)
            distances, indices = tree.query(coordinates, k=self.n_neighbors+1)
            
            # Build sparse matrix
            row_idx = []
            col_idx = []
            weights = []
            
            for i in range(n):
                # Skip self
                for j in range(1, self.n_neighbors+1):
                    if indices[i, j] < n:  # Valid neighbor
                        row_idx.append(i)
                        col_idx.append(indices[i, j])
                        # Inverse distance weight
                        w = 1.0 / (distances[i, j] + 1e-10)
                        weights.append(w)
            
            W = csr_matrix((weights, (row_idx, col_idx)), shape=(n, n))
            
        else:  # distance-based
            coords_km = self._latlon_to_km(coordinates)
            tree = cKDTree(coords_km)
            
            # Find all pairs within threshold
            pairs = tree.query_pairs(self.distance_threshold, output_type='ndarray')
            
            # Build weights
            weights = []
            for i, j in pairs:
                dist = np.linalg.norm(coords_km[i] - coords_km[j])
                w = 1.0 / (dist + 1e-10)
                weights.append(w)
            
            # Create symmetric matrix
            row_idx = np.concatenate([pairs[:, 0], pairs[:, 1]])
            col_idx = np.concatenate([pairs[:, 1], pairs[:, 0]])
            weights = np.concatenate([weights, weights])
            
            W = csr_matrix((weights, (row_idx, col_idx)), shape=(n, n))
        
        # Row-normalize
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        D_inv = csr_matrix((1/row_sums, (range(n), range(n))), shape=(n, n))
        W = D_inv @ W
        
        return W
    
    def _latlon_to_km(self, coordinates: np.ndarray) -> np.ndarray:
        """Convert lat/lon to approximate km using equirectangular projection."""
        lat, lon = coordinates[:, 0], coordinates[:, 1]
        
        # Constants
        R = 6371  # Earth radius in km
        
        # Convert to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Reference latitude (mean)
        lat0 = np.mean(lat_rad)
        
        # Equirectangular projection
        x = R * lon_rad * np.cos(lat0)
        y = R * lat_rad
        
        return np.column_stack([x, y])
    
    def calculate_global_morans(self, 
                                data: np.ndarray,
                                coordinates: np.ndarray,
                                feature_names: List[str] = None) -> Dict[str, float]:
        """Calculate global Moran's I for each feature.
        
        Returns dict mapping feature name to Moran's I value.
        """
        W = self._build_spatial_weights(coordinates)
        n = data.shape[0]
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(data.shape[1])]
        
        results = {}
        
        for i, name in enumerate(feature_names):
            # Skip if all values are missing
            valid_mask = ~np.isnan(data[:, i])
            if not np.any(valid_mask):
                results[name] = np.nan
                continue
            
            # Get valid values
            x = data[:, i].copy()
            x[~valid_mask] = 0
            
            # Standardize
            x_mean = np.mean(x[valid_mask])
            x_centered = x - x_mean
            
            # Calculate Moran's I
            numerator = x_centered @ W @ x_centered
            denominator = np.sum(x_centered ** 2)
            
            if denominator > 0:
                I = (n / W.sum()) * (numerator / denominator)
                results[name] = I
            else:
                results[name] = np.nan
                
        return results