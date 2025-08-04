"""Tensor preprocessing for biodiversity data.

This module provides preprocessing specifically designed for tensor factorization,
avoiding compositional transforms that would break tensor structure.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
import logging
from scipy import stats
from scipy.sparse import coo_matrix

from src.abstractions.interfaces.tensor_analyzer import ITensorPreprocessor
from src.abstractions.types.tensor_types import TensorData

logger = logging.getLogger(__name__)


class TensorPreprocessor(ITensorPreprocessor):
    """Preprocessor for converting biodiversity data to tensor format.
    
    Key features:
    - Grids irregular spatial data
    - Handles extreme sparsity (>95% missing)
    - Simple transformations that preserve tensor structure
    - Mode-specific normalization
    """
    
    def __init__(self, 
                 missing_value: float = np.nan,
                 min_observations_per_cell: int = 1,
                 aggregation_method: str = 'sum'):
        """Initialize tensor preprocessor.
        
        Args:
            missing_value: Value to use for missing data
            min_observations_per_cell: Minimum observations to consider cell valid
            aggregation_method: How to aggregate multiple observations ('sum', 'mean', 'max')
        """
        self.missing_value = missing_value
        self.min_observations_per_cell = min_observations_per_cell
        self.aggregation_method = aggregation_method
        
    def grid_spatial_data(self, coordinates: np.ndarray, values: np.ndarray,
                         lat_bins: int, lon_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """Grid irregular spatial data into regular tensor format.
        
        Args:
            coordinates: (N, 2) array of lat/lon coordinates
            values: (N, M) array of feature values
            lat_bins: Number of latitude bins
            lon_bins: Number of longitude bins
            
        Returns:
            Tuple of (gridded_tensor, grid_coordinates)
        """
        logger.info(f"Gridding {len(coordinates)} points into {lat_bins}x{lon_bins} grid")
        
        # Create grid bounds
        lat_min, lat_max = coordinates[:, 0].min(), coordinates[:, 0].max()
        lon_min, lon_max = coordinates[:, 1].min(), coordinates[:, 1].max()
        
        # Add small buffer to ensure all points fall within grid
        lat_buffer = (lat_max - lat_min) * 0.001
        lon_buffer = (lon_max - lon_min) * 0.001
        
        lat_edges = np.linspace(lat_min - lat_buffer, lat_max + lat_buffer, lat_bins + 1)
        lon_edges = np.linspace(lon_min - lon_buffer, lon_max + lon_buffer, lon_bins + 1)
        
        # Initialize tensor
        n_features = values.shape[1]
        tensor = np.full((lat_bins, lon_bins, n_features), self.missing_value)
        counts = np.zeros((lat_bins, lon_bins))
        
        # Bin coordinates
        lat_indices = np.digitize(coordinates[:, 0], lat_edges) - 1
        lon_indices = np.digitize(coordinates[:, 1], lon_edges) - 1
        
        # Clip to valid range (handles edge cases)
        lat_indices = np.clip(lat_indices, 0, lat_bins - 1)
        lon_indices = np.clip(lon_indices, 0, lon_bins - 1)
        
        # Aggregate values into grid cells
        for i in range(len(coordinates)):
            lat_idx = lat_indices[i]
            lon_idx = lon_indices[i]
            
            # Skip if values are all NaN
            if np.all(np.isnan(values[i])):
                continue
                
            if np.isnan(tensor[lat_idx, lon_idx, 0]):
                # First value in cell
                tensor[lat_idx, lon_idx] = values[i]
                counts[lat_idx, lon_idx] = 1
            else:
                # Aggregate with existing values
                if self.aggregation_method == 'sum':
                    # For sum, we need to handle NaN carefully
                    for j in range(n_features):
                        if not np.isnan(values[i, j]):
                            if np.isnan(tensor[lat_idx, lon_idx, j]):
                                tensor[lat_idx, lon_idx, j] = values[i, j]
                            else:
                                tensor[lat_idx, lon_idx, j] += values[i, j]
                elif self.aggregation_method == 'mean':
                    # Track for mean calculation later
                    for j in range(n_features):
                        if not np.isnan(values[i, j]):
                            if np.isnan(tensor[lat_idx, lon_idx, j]):
                                tensor[lat_idx, lon_idx, j] = values[i, j]
                            else:
                                tensor[lat_idx, lon_idx, j] += values[i, j]
                elif self.aggregation_method == 'max':
                    tensor[lat_idx, lon_idx] = np.nanmax([tensor[lat_idx, lon_idx], values[i]], axis=0)
                
                counts[lat_idx, lon_idx] += 1
        
        # Apply mean if needed
        if self.aggregation_method == 'mean':
            for i in range(lat_bins):
                for j in range(lon_bins):
                    if counts[i, j] > 1:
                        mask = ~np.isnan(tensor[i, j])
                        tensor[i, j, mask] /= counts[i, j]
        
        # Create grid coordinates (cell centers)
        lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
        lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
        
        grid_coords = []
        for i in range(lat_bins):
            for j in range(lon_bins):
                grid_coords.append((lat_centers[i], lon_centers[j]))
        
        # Log sparsity
        n_valid = np.sum(counts >= self.min_observations_per_cell)
        sparsity = 1.0 - (n_valid / (lat_bins * lon_bins))
        logger.info(f"Grid sparsity: {sparsity:.2%} ({n_valid} valid cells)")
        
        return tensor, np.array(grid_coords)
    
    def create_weight_mask(self, tensor: np.ndarray) -> np.ndarray:
        """Create weight mask for missing values.
        
        Args:
            tensor: Input tensor (may contain NaN)
            
        Returns:
            Binary weight mask (1 for observed, 0 for missing)
        """
        # Simple approach: 1 where not NaN, 0 where NaN
        weights = (~np.isnan(tensor)).astype(np.float32)
        
        # Log statistics
        n_observed = np.sum(weights)
        n_total = tensor.size
        logger.info(f"Weight mask: {n_observed}/{n_total} observed ({n_observed/n_total:.2%})")
        
        return weights
    
    def transform_features(self, tensor: np.ndarray, 
                          method: str = 'log1p') -> np.ndarray:
        """Apply feature transformation suitable for tensor factorization.
        
        Args:
            tensor: Input tensor
            method: Transformation method ('log1p', 'sqrt', 'none')
            
        Returns:
            Transformed tensor
        """
        logger.info(f"Applying {method} transformation")
        
        # Create copy to avoid modifying input
        transformed = tensor.copy()
        
        if method == 'log1p':
            # log(x + 1) transformation - handles zeros well
            # Only apply to non-negative values
            mask = ~np.isnan(transformed) & (transformed >= 0)
            transformed[mask] = np.log1p(transformed[mask])
            
        elif method == 'sqrt':
            # Square root transformation
            mask = ~np.isnan(transformed) & (transformed >= 0)
            transformed[mask] = np.sqrt(transformed[mask])
            
        elif method == 'asinh':
            # Inverse hyperbolic sine - handles negative values
            mask = ~np.isnan(transformed)
            transformed[mask] = np.arcsinh(transformed[mask])
            
        elif method == 'none':
            # No transformation
            pass
            
        else:
            raise ValueError(f"Unknown transformation method: {method}")
        
        return transformed
    
    def normalize_modes(self, tensor: np.ndarray, 
                       center: bool = True,
                       scale: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Normalize tensor modes (dimensions).
        
        Args:
            tensor: Input tensor
            center: Whether to center each mode
            scale: Whether to scale each mode
            
        Returns:
            Tuple of (normalized_tensor, normalization_params)
        """
        logger.info(f"Normalizing modes (center={center}, scale={scale})")
        
        normalized = tensor.copy()
        params = {
            'center': center,
            'scale': scale,
            'means': {},
            'stds': {}
        }
        
        # Normalize each mode separately
        n_modes = tensor.ndim
        
        for mode in range(n_modes):
            # Move mode to first axis for easier processing
            perm = list(range(n_modes))
            perm[0], perm[mode] = perm[mode], perm[0]
            tensor_perm = np.transpose(normalized, perm)
            
            # Flatten all other dimensions
            mode_size = tensor_perm.shape[0]
            tensor_flat = tensor_perm.reshape(mode_size, -1)
            
            # Calculate statistics ignoring NaN
            if center:
                means = np.nanmean(tensor_flat, axis=1)
                params['means'][f'mode_{mode}'] = means
                tensor_flat -= means[:, np.newaxis]
            
            if scale:
                stds = np.nanstd(tensor_flat, axis=1)
                # Avoid division by zero
                stds[stds == 0] = 1.0
                params['stds'][f'mode_{mode}'] = stds
                tensor_flat /= stds[:, np.newaxis]
            
            # Reshape back and transpose to original order
            tensor_perm = tensor_flat.reshape(tensor_perm.shape)
            normalized = np.transpose(tensor_perm, np.argsort(perm))
        
        return normalized, params
    
    def create_sparse_tensor(self, tensor: np.ndarray, 
                           threshold: float = 0.95) -> Optional[Dict[str, Any]]:
        """Convert to sparse format if sparsity exceeds threshold.
        
        Args:
            tensor: Input tensor
            threshold: Sparsity threshold for conversion
            
        Returns:
            Sparse representation or None if not sparse enough
        """
        # Calculate sparsity
        n_missing = np.sum(np.isnan(tensor))
        n_zeros = np.sum(tensor == 0, where=~np.isnan(tensor))
        sparsity = (n_missing + n_zeros) / tensor.size
        
        if sparsity < threshold:
            logger.info(f"Sparsity {sparsity:.2%} below threshold {threshold:.2%}, keeping dense")
            return None
        
        logger.info(f"Converting to sparse format (sparsity: {sparsity:.2%})")
        
        # For 3D tensor, we need to handle this carefully
        # Store as dictionary of 2D sparse matrices (one per feature)
        sparse_data = {}
        
        if tensor.ndim == 3:
            n_features = tensor.shape[2]
            for f in range(n_features):
                # Get 2D slice
                slice_2d = tensor[:, :, f]
                
                # Replace NaN with 0 for sparse format
                slice_clean = np.nan_to_num(slice_2d, nan=0.0)
                
                # Convert to COO format
                sparse_data[f'feature_{f}'] = coo_matrix(slice_clean)
        else:
            raise NotImplementedError("Only 3D tensors supported for sparse conversion")
        
        return {
            'sparse_matrices': sparse_data,
            'shape': tensor.shape,
            'sparsity': sparsity,
            'format': 'coo'
        }
    
    def dataframe_to_tensor(self, df: pd.DataFrame, 
                           lat_bins: int = 20,
                           lon_bins: int = 20,
                           transform: str = 'log1p') -> TensorData:
        """Convert a dataframe to tensor format.
        
        Args:
            df: Input dataframe with columns [latitude, longitude, features...]
            lat_bins: Number of latitude bins for gridding
            lon_bins: Number of longitude bins for gridding
            transform: Feature transformation method
            
        Returns:
            TensorData object ready for factorization
        """
        logger.info(f"Converting dataframe with {len(df)} rows to tensor")
        
        # Extract coordinates
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            raise ValueError("Dataframe must have 'latitude' and 'longitude' columns")
        
        coordinates = df[['latitude', 'longitude']].values
        
        # Extract feature columns (everything except lat/lon)
        feature_cols = [col for col in df.columns if col not in ['latitude', 'longitude']]
        if not feature_cols:
            raise ValueError("No feature columns found in dataframe")
        
        values = df[feature_cols].values
        
        # Grid the data
        tensor, grid_coords = self.grid_spatial_data(
            coordinates, values, lat_bins, lon_bins
        )
        
        # Apply transformation
        if transform != 'none':
            tensor = self.transform_features(tensor, method=transform)
        
        # Create weight mask
        weights = self.create_weight_mask(tensor)
        
        # Calculate sparsity
        n_missing = np.sum(np.isnan(tensor))
        n_zeros = np.sum(tensor == 0, where=~np.isnan(tensor))
        sparsity = (n_missing + n_zeros) / tensor.size
        
        # Create TensorData object
        return TensorData(
            tensor=tensor,
            weights=weights,
            coordinates=[],  # Grid-based, so coordinates are implicit
            feature_names=feature_cols,
            metadata={
                'mode_names': ['latitude', 'longitude', 'features'],
                'lat_bins': lat_bins,
                'lon_bins': lon_bins,
                'transform': transform,
                'aggregation': self.aggregation_method,
                'sparsity': sparsity
            },
            is_sparse=False
        )


def create_biodiversity_tensor(data_df: pd.DataFrame,
                              lat_col: str,
                              lon_col: str,
                              feature_cols: List[str],
                              lat_bins: int = 50,
                              lon_bins: int = 50,
                              transform: str = 'log1p',
                              normalize: bool = True) -> TensorData:
    """Convenience function to create tensor from biodiversity dataframe.
    
    Args:
        data_df: Input dataframe
        lat_col: Name of latitude column
        lon_col: Name of longitude column  
        feature_cols: List of feature column names
        lat_bins: Number of latitude bins
        lon_bins: Number of longitude bins
        transform: Feature transformation method
        normalize: Whether to normalize modes
        
    Returns:
        TensorData object ready for factorization
    """
    # Extract coordinates and values
    coordinates = data_df[[lat_col, lon_col]].values
    values = data_df[feature_cols].values
    
    # Create preprocessor
    preprocessor = TensorPreprocessor(aggregation_method='sum')
    
    # Grid the data
    tensor, grid_coords = preprocessor.grid_spatial_data(
        coordinates, values, lat_bins, lon_bins
    )
    
    # Transform features
    tensor = preprocessor.transform_features(tensor, method=transform)
    
    # Normalize if requested
    normalization_params = None
    if normalize:
        tensor, normalization_params = preprocessor.normalize_modes(tensor)
    
    # Create weight mask
    weights = preprocessor.create_weight_mask(tensor)
    
    # Create metadata
    metadata = {
        'lat_bins': lat_bins,
        'lon_bins': lon_bins,
        'transform': transform,
        'normalized': normalize,
        'normalization_params': normalization_params,
        'aggregation_method': 'sum',
        'original_shape': data_df.shape
    }
    
    # Check if sparse representation would be beneficial
    sparsity = 1.0 - (np.sum(weights) / weights.size)
    is_sparse = sparsity > 0.95
    
    return TensorData(
        tensor=tensor,
        weights=weights,
        coordinates=grid_coords.tolist(),
        feature_names=feature_cols,
        metadata=metadata,
        is_sparse=is_sparse
    )