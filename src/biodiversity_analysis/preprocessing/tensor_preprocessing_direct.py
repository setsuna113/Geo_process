from src.abstractions import get_logger
"""Direct tensor preprocessing for biodiversity data WITHOUT GRIDDING.

The parquet data is ALREADY a sparse tensor - we just need to format it properly!
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, List, Union, Dict, Any
from scipy import sparse

from src.abstractions import TensorData

logger = get_logger(__name__)


class TensorPreprocessor:
    """Direct preprocessor for converting parquet data to tensor format.
    
    NO GRIDDING - the data is already at its native resolution!
    Each row in the parquet is a tensor element with coordinates and values.
    """
    
    def __init__(self, 
                 missing_value: float = 0.0,
                 min_observations_per_cell: int = 1):
        """Initialize tensor preprocessor.
        
        Args:
            missing_value: Value to use for missing data (typically 0 for sparse)
            min_observations_per_cell: Minimum observations required (not used for direct conversion)
        """
        self.missing_value = missing_value
        self.min_observations_per_cell = min_observations_per_cell
    
    def parquet_to_tensor(self, df: pd.DataFrame,
                          coordinate_cols: List[str] = None,
                          feature_cols: List[str] = None,
                          transform: str = 'none') -> TensorData:
        """Convert parquet dataframe DIRECTLY to sparse tensor representation.
        
        The parquet IS the tensor - each row has coordinates and values.
        We just need to convert to the right format for CP-WOPT.
        
        Args:
            df: DataFrame from parquet with coordinates and features
            coordinate_cols: Column names for coordinates (default: ['x', 'y'])
            feature_cols: Feature columns to use (auto-detect if None)
            transform: Transformation to apply ('none', 'log1p', 'sqrt', 'asinh')
            
        Returns:
            TensorData in sparse format ready for HiCOO conversion
        """
        # Default coordinate columns
        if coordinate_cols is None:
            # Try common coordinate column names
            if 'x' in df.columns and 'y' in df.columns:
                coordinate_cols = ['x', 'y']
            elif 'longitude' in df.columns and 'latitude' in df.columns:
                coordinate_cols = ['longitude', 'latitude']
            elif 'lon' in df.columns and 'lat' in df.columns:
                coordinate_cols = ['lon', 'lat']
            else:
                raise ValueError("Could not auto-detect coordinate columns")
        
        x_col, y_col = coordinate_cols
        
        # Get feature columns
        if feature_cols is None:
            # Auto-detect numeric columns that aren't coordinates
            numeric_dtypes = [np.float32, np.float64, np.int32, np.int64, 'float32', 'float64', 'int32', 'int64']
            feature_cols = [col for col in df.columns 
                          if col not in coordinate_cols and 
                          df[col].dtype in numeric_dtypes]
            logger.info(f"Auto-detected {len(feature_cols)} feature columns")
        
        # Get coordinates
        x_coords = df[x_col].values
        y_coords = df[y_col].values
        
        # Get unique coordinates to determine tensor dimensions
        x_unique = np.unique(x_coords)
        y_unique = np.unique(y_coords)
        
        logger.info(f"Tensor dimensions: {len(y_unique)} x {len(x_unique)} x {len(feature_cols)}")
        logger.info(f"Data points: {len(df):,}")
        
        # Create coordinate to index mappings
        x_to_idx = {float(x): i for i, x in enumerate(x_unique)}
        y_to_idx = {float(y): i for i, y in enumerate(y_unique)}
        
        # Convert coordinates to indices
        x_indices = np.array([x_to_idx[float(x)] for x in x_coords], dtype=np.int32)
        y_indices = np.array([y_to_idx[float(y)] for y in y_coords], dtype=np.int32)
        
        # Extract and transform feature values
        values = df[feature_cols].values.astype(np.float32)
        
        # Apply transformation
        if transform == 'log1p':
            values = np.log1p(np.maximum(values, 0))
        elif transform == 'sqrt':
            values = np.sqrt(np.maximum(values, 0))
        elif transform == 'asinh':
            values = np.arcsinh(values)
        elif transform != 'none':
            logger.warning(f"Unknown transform '{transform}', using 'none'")
        
        # Tensor shape
        shape = (len(y_unique), len(x_unique), len(feature_cols))
        
        # Build COO format sparse representation
        # This will be converted to HiCOO by the sparse handler
        row_indices = []
        col_indices = []
        depth_indices = []
        data_values = []
        
        for i in range(len(df)):
            for j, feat in enumerate(feature_cols):
                val = values[i, j]
                if not np.isnan(val) and val != 0:
                    row_indices.append(y_indices[i])
                    col_indices.append(x_indices[i])
                    depth_indices.append(j)
                    data_values.append(val)
        
        # Convert to numpy arrays
        indices = np.array([row_indices, col_indices, depth_indices], dtype=np.int32).T
        data_values = np.array(data_values, dtype=np.float32)
        
        # Calculate sparsity
        total_elements = np.prod(shape)
        nnz = len(data_values)
        sparsity = 1.0 - (nnz / total_elements) if total_elements > 0 else 1.0
        
        logger.info(f"Sparse tensor created: shape={shape}, nnz={nnz:,}, sparsity={sparsity:.1%}")
        logger.info("NO GRIDDING performed - using data at native resolution!")
        
        # Create dense tensor - always create full tensor for consistency
        tensor = np.zeros(shape, dtype=np.float32)
        weights = np.zeros(shape, dtype=np.float32)  # Start with zeros for weights
        
        # Fill in the observed values
        for idx, val in zip(indices, data_values):
            i, j, k = idx
            tensor[i, j, k] = val
            weights[i, j, k] = 1.0  # Mark as observed
        
        # Store sparse representation in metadata for efficiency
        sparse_data = {
            'indices': indices,
            'values': data_values,
            'shape': shape,
            'nnz': nnz,
            'sparsity': sparsity
        }
        
        # Create full coordinate list - TensorData expects this
        coord_list = []
        for i, y in enumerate(y_unique):
            for j, x in enumerate(x_unique):
                coord_list.append((y, x))  # (lat, lon) format
        
        # Return TensorData with proper structure
        return TensorData(
            tensor=tensor,
            weights=weights,
            coordinates=coord_list,  # Full coordinate list required
            feature_names=feature_cols,
            metadata={
                'source': 'parquet_direct',
                'n_points': len(df),
                'transform': transform,
                'no_gridding': True,
                'coordinate_system': 'geographic',
                'actual_shape': shape,
                'actual_sparsity': sparsity,
                'sparse_data': sparse_data,  # Store sparse representation
                'x_coords': x_unique.tolist(),
                'y_coords': y_unique.tolist()
            },
            is_sparse=sparsity > 0.5
        )
