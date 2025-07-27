# src/processors/spatial_analysis/data_processor.py
"""Data processing utilities for spatial analysis."""

import logging
from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np
import xarray as xr
from pathlib import Path

from src.infrastructure.processors.base_processor import EnhancedBaseProcessor as BaseProcessor
from src.config import config
from src.config.config import Config
from src.database.connection import DatabaseManager
from src.processors.data_preparation.array_converter import ArrayConverter
from src.spatial_analysis.memory_aware_processor import MemoryAwareProcessor, check_memory_usage

logger = logging.getLogger(__name__)

class SpatialDataProcessor(BaseProcessor):
    """Handle data preparation and processing for spatial analysis."""
    
    def __init__(self, config: Config, **kwargs):
        """Initialize processor with configuration."""
        # Extract processor-specific parameters from config
        batch_size = config.get('spatial_analysis', {}).get('batch_size', 1000)
        max_workers = config.get('spatial_analysis', {}).get('max_workers', None)
        memory_limit_mb = config.get('spatial_analysis', {}).get('memory_limit_mb', None)
        
        super().__init__(
            batch_size=batch_size,
            max_workers=max_workers,
            memory_limit_mb=memory_limit_mb,
            **kwargs
        )
        
        self.config = config
        self.array_converter = ArrayConverter(config)
        self.memory_processor = MemoryAwareProcessor(
            memory_limit_gb=config.get('processing', {}).get('subsampling', {}).get('memory_limit_gb', 8.0)
        )
        
        # Processing parameters
        self.normalize_data = config.get('spatial_analysis', {}).get('normalize_data', True)
    
    def prepare_data(self, 
                    data: Union[xr.Dataset, xr.DataArray, np.ndarray],
                    normalize: Optional[bool] = None,
                    flatten: bool = True,
                    memory_aware: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Prepare data for analysis with memory-aware processing.
        
        Args:
            data: Input data in various formats
            normalize: Whether to normalize (uses config default if None)
            flatten: Whether to flatten spatial dimensions
            memory_aware: Whether to use memory-aware processing
            
        Returns:
            Tuple of (prepared_array, metadata_dict)
        """
        logger.info(f"Preparing data with memory-aware processing: {memory_aware}")
        
        # Convert to xarray if needed
        if isinstance(data, np.ndarray):
            if data.ndim == 3:
                # Assume shape is (bands, height, width)
                data = xr.DataArray(
                    data,
                    dims=['band', 'y', 'x'],
                    coords={'band': [f'band_{i}' for i in range(data.shape[0])]}
                )
            else:
                raise ValueError("Numpy array must be 3D (bands, height, width)")
        
        # Store original metadata
        if isinstance(data, xr.Dataset):
            # For Dataset, use shape of first data variable for spatial dims
            first_var = list(data.data_vars)[0]
            spatial_shape = data[first_var].shape
            # But for full shape, include all variables as bands
            data_shape = (len(data.data_vars),) + spatial_shape
        else:
            data_shape = data.shape
            
        metadata = {
            'original_type': type(data).__name__,
            'original_shape': data_shape,
            'dims': list(data.dims) if hasattr(data, 'dims') else None,
            'coords': {k: v.values.tolist() for k, v in data.coords.items()} if hasattr(data, 'coords') else None
        }
        
        # Extract band names
        if isinstance(data, xr.Dataset):
            metadata['bands'] = list(data.data_vars)
        elif isinstance(data, xr.DataArray) and 'band' in data.dims:
            metadata['bands'] = data.coords['band'].values.tolist()
        else:
            metadata['bands'] = ['value']
        
        # Check memory requirements if memory_aware is enabled
        if memory_aware:
            # Get appropriate dtype for memory calculation
            if isinstance(data, xr.Dataset):
                first_var = list(data.data_vars)[0]
                data_dtype = data[first_var].dtype
            elif isinstance(data, xr.DataArray):
                data_dtype = data.dtype
            elif hasattr(data, 'dtype'):
                data_dtype = data.dtype
            else:
                data_dtype = np.float64
                
            memory_info = check_memory_usage(data_shape, data_dtype)
            logger.info(f"Data size: {memory_info['data_size_gb']:.2f} GB, "
                       f"Available: {memory_info['available_gb']:.2f} GB")
            
            if not memory_info['will_fit']:
                logger.warning("Dataset may not fit in memory, consider using chunked processing")
                # Here we could implement chunked processing in the future
                # For now, we'll proceed but warn the user
        
        # Normalize if requested
        if normalize is None:
            normalize = self.normalize_data
            
        if normalize:
            logger.info("Normalizing data")
            # For now, simple normalization - could be moved to a separate normalizer
            if isinstance(data, (xr.Dataset, xr.DataArray)):
                data = (data - data.mean()) / data.std()
            metadata['normalized'] = True
        else:
            metadata['normalized'] = False
        
        # Convert to numpy for analysis
        if flatten:
            # Flatten spatial dimensions while preserving band structure
            if isinstance(data, (xr.Dataset, xr.DataArray)):
                result = self.array_converter.xarray_to_numpy(data, flatten=True, preserve_coords=True)
                prepared_array = result['array']
                metadata.update(result)
            else:
                # For numpy arrays, handle flattening manually
                prepared_array = data.flatten()
                metadata['coords_info'] = None
            
            # Reshape to (n_samples, n_features) for sklearn-like interface
            if prepared_array.ndim == 1:
                prepared_array = prepared_array.reshape(-1, 1)
        else:
            # Keep spatial structure
            if isinstance(data, xr.Dataset):
                # Stack variables as bands
                prepared_array = np.stack([data[var].values for var in data.data_vars])
            elif isinstance(data, xr.DataArray):
                prepared_array = data.values
            else:
                # data is already a numpy array
                prepared_array = data
            
            metadata['spatial_shape'] = prepared_array.shape
        
        metadata['prepared_shape'] = prepared_array.shape
        logger.info(f"Prepared data shape: {prepared_array.shape}")
        
        return prepared_array, metadata
    
    def restore_spatial_structure(self, labels: np.ndarray, 
                                 metadata: Dict[str, Any]) -> Optional[Union[xr.Dataset, xr.DataArray]]:
        """
        Restore spatial structure to flat analysis results.
        
        Args:
            labels: Flat array of cluster/region assignments
            metadata: Metadata from prepare_data
            
        Returns:
            Spatially structured results or None if not possible
        """
        if 'coords_info' not in metadata or metadata['coords_info'] is None:
            return None
        
        try:
            # Use array converter to restore structure
            spatial_output = self.array_converter.numpy_to_xarray(
                labels, 
                metadata['coords_info'], 
                metadata.get('dims', ['y', 'x'])
            )
            
            # Add meaningful variable name
            if isinstance(spatial_output, xr.DataArray):
                spatial_output.name = 'cluster_labels'
                spatial_output.attrs['long_name'] = 'Spatial cluster assignments'
                spatial_output.attrs['description'] = 'Cluster labels from spatial analysis'
            
            return spatial_output
            
        except Exception as e:
            logger.warning(f"Could not restore spatial structure: {e}")
            return None
    
    def estimate_memory_requirements(self, data_shape: Tuple[int, ...], 
                                   dtype: np.dtype = np.dtype(np.float64)) -> Dict[str, Any]:
        """Estimate memory requirements for processing."""
        return check_memory_usage(data_shape, dtype)
    
    def validate_input_data(self, data: Union[xr.Dataset, xr.DataArray, np.ndarray]) -> Tuple[bool, List[str]]:
        """
        Validate input data for analysis.
        
        Args:
            data: Input data to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check data type
        if not isinstance(data, (xr.Dataset, xr.DataArray, np.ndarray)):
            issues.append(f"Unsupported data type: {type(data)}")
            return False, issues
        
        # Check for empty data
        if isinstance(data, np.ndarray):
            if data.size == 0:
                issues.append("Data array is empty")
        elif isinstance(data, xr.Dataset):
            # For Dataset, check if any data variables are empty
            if len(data.data_vars) == 0:
                issues.append("Dataset has no data variables")
            else:
                first_var = list(data.data_vars)[0]
                if data[first_var].size == 0:
                    issues.append("Data array is empty")
        elif isinstance(data, xr.DataArray):
            if data.size == 0:
                issues.append("Data array is empty")
        
        # Check for all NaN/null values
        if isinstance(data, np.ndarray):
            if np.all(np.isnan(data)):
                issues.append("All data values are NaN")
        elif isinstance(data, xr.Dataset):
            # Check first data variable
            first_var = list(data.data_vars)[0]
            if data[first_var].isnull().all():
                issues.append("All data values are null/NaN")
        elif isinstance(data, xr.DataArray):
            if data.isnull().all():
                issues.append("All data values are null/NaN")
        
        # Check dimensions
        if isinstance(data, np.ndarray):
            if data.ndim < 2:
                issues.append("Data must be at least 2-dimensional")
            elif data.ndim > 4:
                issues.append("Data has too many dimensions (max 4)")
        
        return len(issues) == 0, issues
    
    def process_single(self, item: Any) -> Any:
        """Process a single item (required by BaseProcessor)."""
        # For spatial data processor, this is handled by prepare_data
        return item
    
    def validate_input(self, data: Any) -> bool:
        """Validate input (required by BaseProcessor)."""
        valid, _ = self.validate_input_data(data)
        return valid