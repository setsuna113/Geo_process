"""Tensor preprocessing utilities for biodiversity data.

This module provides utilities for converting biodiversity data into tensor format
suitable for CP-WOPT decomposition.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler

from src.abstractions import TensorData

logger = logging.getLogger(__name__)


class TensorPreprocessor:
    """Preprocessor for converting biodiversity data to tensor format."""
    
    def __init__(self, 
                 missing_value: float = np.nan,
                 min_observations_per_cell: int = 1,
                 aggregation_method: str = 'mean'):
        """Initialize tensor preprocessor.
        
        NOTE: Sampling is now handled exclusively by ParquetLoader:
        - Use spatial_grid_step for spatial grid sampling (every Nth point)
        - Use sample_size for random sampling (N random points)
        
        Args:
            missing_value: Value to use for missing data
            min_observations_per_cell: Minimum observations required per grid cell
            aggregation_method: How to aggregate multiple observations ('mean', 'sum', 'max')
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
            Tuple of (gridded_tensor, weight_mask)
        """
        if coordinates.shape[0] != values.shape[0]:
            raise ValueError(f"Coordinate rows {coordinates.shape[0]} != value rows {values.shape[0]}")
        
        # Create bin edges
        lat_min, lat_max = coordinates[:, 0].min(), coordinates[:, 0].max()
        lon_min, lon_max = coordinates[:, 1].min(), coordinates[:, 1].max()
        
        lat_edges = np.linspace(lat_min, lat_max, lat_bins + 1)
        lon_edges = np.linspace(lon_min, lon_max, lon_bins + 1)
        
        # Initialize tensor
        n_features = values.shape[1] if values.ndim > 1 else 1
        tensor = np.full((lat_bins, lon_bins, n_features), self.missing_value)
        weights = np.zeros((lat_bins, lon_bins, n_features))
        counts = np.zeros((lat_bins, lon_bins, n_features))
        
        # Bin the data
        lat_indices = np.digitize(coordinates[:, 0], lat_edges) - 1
        lon_indices = np.digitize(coordinates[:, 1], lon_edges) - 1
        
        # Clip to valid range
        lat_indices = np.clip(lat_indices, 0, lat_bins - 1)
        lon_indices = np.clip(lon_indices, 0, lon_bins - 1)
        
        # Aggregate values
        for i in range(len(coordinates)):
            lat_idx = lat_indices[i]
            lon_idx = lon_indices[i]
            
            if values.ndim == 1:
                val = values[i]
                if not np.isnan(val):
                    if counts[lat_idx, lon_idx, 0] == 0:
                        tensor[lat_idx, lon_idx, 0] = val
                    else:
                        if self.aggregation_method == 'mean':
                            # Running mean
                            n = counts[lat_idx, lon_idx, 0]
                            tensor[lat_idx, lon_idx, 0] = (tensor[lat_idx, lon_idx, 0] * n + val) / (n + 1)
                        elif self.aggregation_method == 'sum':
                            tensor[lat_idx, lon_idx, 0] += val
                        elif self.aggregation_method == 'max':
                            tensor[lat_idx, lon_idx, 0] = max(tensor[lat_idx, lon_idx, 0], val)
                    
                    counts[lat_idx, lon_idx, 0] += 1
                    weights[lat_idx, lon_idx, 0] = 1
            else:
                for j in range(n_features):
                    val = values[i, j]
                    if not np.isnan(val):
                        if counts[lat_idx, lon_idx, j] == 0:
                            tensor[lat_idx, lon_idx, j] = val
                        else:
                            if self.aggregation_method == 'mean':
                                n = counts[lat_idx, lon_idx, j]
                                tensor[lat_idx, lon_idx, j] = (tensor[lat_idx, lon_idx, j] * n + val) / (n + 1)
                            elif self.aggregation_method == 'sum':
                                tensor[lat_idx, lon_idx, j] += val
                            elif self.aggregation_method == 'max':
                                tensor[lat_idx, lon_idx, j] = max(tensor[lat_idx, lon_idx, j], val)
                        
                        counts[lat_idx, lon_idx, j] += 1
                        weights[lat_idx, lon_idx, j] = 1
        
        # Apply minimum observation threshold
        insufficient_mask = counts < self.min_observations_per_cell
        tensor[insufficient_mask] = self.missing_value
        weights[insufficient_mask] = 0
        
        # Replace missing values with 0 for CP decomposition
        tensor[np.isnan(tensor)] = 0
        
        logger.info(f"Gridded {len(coordinates)} points into {lat_bins}x{lon_bins}x{n_features} tensor")
        logger.info(f"Grid cells with data: {np.sum(weights[:, :, 0] > 0)}/{lat_bins * lon_bins}")
        
        return tensor, weights
    
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
        tensor, weights = self.grid_spatial_data(coordinates, values, lat_bins, lon_bins)
        
        # Apply transformation
        tensor = self.transform_features(tensor, method=transform)
        
        # Create coordinate grid for tensor
        lat_min, lat_max = coordinates[:, 0].min(), coordinates[:, 0].max()
        lon_min, lon_max = coordinates[:, 1].min(), coordinates[:, 1].max()
        
        lat_centers = np.linspace(lat_min, lat_max, lat_bins)
        lon_centers = np.linspace(lon_min, lon_max, lon_bins)
        
        tensor_coords = []
        for lat in lat_centers:
            for lon in lon_centers:
                tensor_coords.append((lat, lon))
        
        # Create TensorData object
        tensor_data = TensorData(
            tensor=tensor,
            weights=weights,
            coordinates=tensor_coords,
            feature_names=feature_cols,
            metadata={
                'lat_bins': lat_bins,
                'lon_bins': lon_bins,
                'transform': transform,
                'aggregation': self.aggregation_method,
                'original_shape': df.shape
            }
        )
        
        return tensor_data
    
    def transform_features(self, tensor: np.ndarray, 
                          method: str = 'log1p') -> np.ndarray:
        """Apply transformation to tensor features.
        
        Args:
            tensor: Input tensor
            method: Transformation method ('log1p', 'sqrt', 'none', 'standardize')
            
        Returns:
            Transformed tensor
        """
        if method == 'none':
            return tensor
        
        # Apply transformation
        if method == 'log1p':
            # Log transform with offset for zeros
            tensor = np.log1p(np.maximum(tensor, 0))
        elif method == 'sqrt':
            # Square root transform
            tensor = np.sqrt(np.maximum(tensor, 0))
        elif method == 'standardize':
            # Standardize each feature
            for i in range(tensor.shape[2]):
                feature_slice = tensor[:, :, i]
                valid_mask = feature_slice != 0
                if np.any(valid_mask):
                    mean = np.mean(feature_slice[valid_mask])
                    std = np.std(feature_slice[valid_mask])
                    if std > 0:
                        tensor[:, :, i] = (feature_slice - mean) / std
        elif method == 'robust':
            # Robust scaling using median and MAD
            for i in range(tensor.shape[2]):
                feature_slice = tensor[:, :, i]
                valid_mask = feature_slice != 0
                if np.any(valid_mask):
                    median = np.median(feature_slice[valid_mask])
                    mad = np.median(np.abs(feature_slice[valid_mask] - median))
                    if mad > 0:
                        tensor[:, :, i] = (feature_slice - median) / (1.4826 * mad)
        else:
            logger.warning(f"Unknown transform method: {method}")
        
        return tensor
    
    def normalize_modes(self, tensor: np.ndarray, 
                       mode_weights: Optional[List[float]] = None) -> np.ndarray:
        """Normalize tensor modes to have similar scales.
        
        Args:
            tensor: Input tensor
            mode_weights: Optional weights for each mode
            
        Returns:
            Normalized tensor
        """
        if mode_weights is None:
            mode_weights = [1.0] * tensor.ndim
        
        # Normalize each mode
        for mode in range(tensor.ndim):
            # Compute mode norm
            mode_norm = np.linalg.norm(tensor, axis=mode, keepdims=True)
            mode_norm[mode_norm == 0] = 1  # Avoid division by zero
            
            # Apply normalization
            tensor = tensor / mode_norm * mode_weights[mode]
        
        return tensor
    
    def create_sparse_tensor(self, coordinates: np.ndarray, 
                           values: np.ndarray,
                           shape: Tuple[int, ...],
                           feature_names: Optional[List[str]] = None) -> TensorData:
        """Create a sparse tensor representation.
        
        Args:
            coordinates: (N, ndim) array of coordinates
            values: (N,) array of values
            shape: Shape of full tensor
            feature_names: Optional feature names
            
        Returns:
            TensorData in sparse format
        """
        # Validate inputs
        if coordinates.shape[0] != values.shape[0]:
            raise ValueError("Number of coordinates must match number of values")
        
        if coordinates.shape[1] != len(shape):
            raise ValueError(f"Coordinate dimensions {coordinates.shape[1]} != tensor dimensions {len(shape)}")
        
        # Create weight mask
        weights = np.ones_like(values)
        
        # Handle missing values
        valid_mask = ~np.isnan(values)
        coordinates = coordinates[valid_mask]
        values = values[valid_mask]
        weights = weights[valid_mask]
        
        # Create sparse tensor data
        tensor_data = TensorData(
            tensor=self._sparse_to_dense(coordinates, values, shape),
            weights=self._sparse_to_dense(coordinates, weights, shape),
            coordinates=[(c[0], c[1]) for c in coordinates[:, :2]] if coordinates.shape[1] >= 2 else [],
            feature_names=feature_names or [f"feature_{i}" for i in range(shape[-1] if len(shape) > 2 else 1)],
            metadata={
                'sparse': True,
                'nnz': len(values),
                'density': len(values) / np.prod(shape)
            }
        )
        
        return tensor_data
    
    def _sparse_to_dense(self, coordinates: np.ndarray, 
                        values: np.ndarray, 
                        shape: Tuple[int, ...]) -> np.ndarray:
        """Convert sparse representation to dense tensor.
        
        Args:
            coordinates: Sparse coordinates
            values: Sparse values
            shape: Dense tensor shape
            
        Returns:
            Dense tensor
        """
        dense = np.zeros(shape)
        for i, coord in enumerate(coordinates):
            dense[tuple(coord)] = values[i]
        return dense


def create_biodiversity_tensor(data: pd.DataFrame,
                              lat_bins: int = 20,
                              lon_bins: int = 20,
                              feature_transform: str = 'log1p',
                              aggregation: str = 'mean',
                              min_observations: int = 1) -> TensorData:
    """Convenience function to create biodiversity tensor from dataframe.
    
    Args:
        data: Input dataframe with lat/lon and feature columns
        lat_bins: Number of latitude bins
        lon_bins: Number of longitude bins
        feature_transform: Transformation to apply to features
        aggregation: Aggregation method for multiple observations
        min_observations: Minimum observations per cell
        
    Returns:
        TensorData object ready for CP-WOPT
    """
    preprocessor = TensorPreprocessor(
        missing_value=0.0,
        min_observations_per_cell=min_observations,
        aggregation_method=aggregation
    )
    
    return preprocessor.dataframe_to_tensor(
        data, 
        lat_bins=lat_bins,
        lon_bins=lon_bins,
        transform=feature_transform
    )


def validate_tensor_data(tensor_data: TensorData) -> bool:
    """Validate tensor data for CP-WOPT.
    
    Args:
        tensor_data: TensorData to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    # Check tensor and weights have same shape
    if tensor_data.tensor.shape != tensor_data.weights.shape:
        raise ValueError(f"Tensor shape {tensor_data.tensor.shape} != weights shape {tensor_data.weights.shape}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(tensor_data.tensor)) or np.any(np.isinf(tensor_data.tensor)):
        raise ValueError("Tensor contains NaN or Inf values")
    
    # Check weights are binary
    unique_weights = np.unique(tensor_data.weights)
    if not np.all(np.isin(unique_weights, [0, 1])):
        raise ValueError(f"Weights must be binary (0 or 1), got {unique_weights}")
    
    # Check coordinates if provided
    if tensor_data.coordinates:
        expected_coords = tensor_data.tensor.shape[0] * tensor_data.tensor.shape[1]
        if len(tensor_data.coordinates) != expected_coords:
            logger.warning(f"Coordinate count {len(tensor_data.coordinates)} != expected {expected_coords}")
    
    # Check feature names
    if tensor_data.ndim >= 3:
        if len(tensor_data.feature_names) != tensor_data.tensor.shape[2]:
            raise ValueError(f"Feature names {len(tensor_data.feature_names)} != tensor features {tensor_data.tensor.shape[2]}")
    
    return True


def create_tensor_from_sparse(indices: np.ndarray,
                             values: np.ndarray,
                             shape: Tuple[int, ...],
                             feature_names: Optional[List[str]] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> TensorData:
    """Create TensorData from sparse representation.
    
    Args:
        indices: (nnz, ndim) array of indices
        values: (nnz,) array of values
        shape: Shape of full tensor
        feature_names: Optional feature names
        metadata: Optional metadata
        
    Returns:
        TensorData object
    """
    # Convert to dense
    tensor = np.zeros(shape)
    weights = np.zeros(shape)
    
    for i, (idx, val) in enumerate(zip(indices, values)):
        tensor[tuple(idx)] = val
        weights[tuple(idx)] = 1
    
    # Create coordinate list
    if len(shape) >= 2:
        coords = [(i, j) for i in range(shape[0]) for j in range(shape[1])]
    else:
        coords = []
    
    # Default feature names
    if feature_names is None and len(shape) >= 3:
        feature_names = [f"feature_{i}" for i in range(shape[2])]
    elif feature_names is None:
        feature_names = ["feature_0"]
    
    # Default metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'original_format': 'sparse',
        'nnz': len(values),
        'density': len(values) / np.prod(shape)
    })
    
    # Determine if sparse
    is_sparse = len(values) / np.prod(shape) < 0.1
    
    return TensorData(
        tensor=tensor,
        weights=weights,
        coordinates=coords,
        feature_names=feature_names,
        metadata=metadata,
        is_sparse=is_sparse
    )


# REMOVED TensorPreprocessorWithSampling class
# Sampling is now handled exclusively by ParquetLoader:
#   - spatial_grid_step: For spatial grid sampling (takes every Nth point)
#   - sample_size: For random sampling (takes N random points)
# This is more efficient as it happens during data loading, not after tensor creation