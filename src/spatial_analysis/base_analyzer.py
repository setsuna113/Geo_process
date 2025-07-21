# src/spatial_analysis/base_analyzer.py
"""Abstract base class for spatial analysis methods."""

import logging
from abc import abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import xarray as xr
import geopandas as gpd
from dataclasses import dataclass, asdict

from src.base.processor import BaseProcessor
from src.config.config import Config
from src.database.connection import DatabaseManager
from src.processors.data_preparation.array_converter import ArrayConverter

logger = logging.getLogger(__name__)

@dataclass
class AnalysisMetadata:
    """Metadata for spatial analysis runs."""
    analysis_type: str
    input_shape: Tuple[int, ...]
    input_bands: List[str]
    parameters: Dict[str, Any]
    processing_time: float
    timestamp: str
    data_source: Optional[str] = None
    normalization_applied: bool = False
    coordinate_system: str = "EPSG:4326"

@dataclass
class AnalysisResult:
    """Container for analysis results."""
    labels: np.ndarray  # Cluster/region assignments
    metadata: AnalysisMetadata
    statistics: Dict[str, Any]
    spatial_output: Optional[Union[xr.Dataset, xr.DataArray]] = None
    additional_outputs: Optional[Dict[str, Any]] = None

class BaseAnalyzer(BaseProcessor):
    """
    Abstract base class for spatial analysis methods.
    
    Provides common functionality for:
    - Data loading and preprocessing
    - Result storage and retrieval
    - Progress tracking
    - Database integration
    """
    
    def __init__(self, config: Config, db_connection: Optional[DatabaseManager] = None, **kwargs):
        """
        Initialize analyzer with configuration.
        
        Args:
            config: System configuration
            db_connection: Optional database connection for result storage
            **kwargs: Additional arguments for BaseProcessor
        """
        # Extract processor-specific parameters from config
        batch_size = config.get('spatial_analysis', {}).get('batch_size', 1000)
        max_workers = config.get('spatial_analysis', {}).get('max_workers', None)
        store_results = config.get('spatial_analysis', {}).get('save_results', True)
        memory_limit_mb = config.get('spatial_analysis', {}).get('memory_limit_mb', None)
        
        # Initialize BaseProcessor with proper parameters
        super().__init__(
            batch_size=batch_size,
            max_workers=max_workers,
            store_results=store_results,
            memory_limit_mb=memory_limit_mb,
            **kwargs
        )
        
        self.config = config
        self.db = db_connection
        self.array_converter = ArrayConverter(config)
        # TODO: Fix DataNormalizer abstract method implementation
        self.normalizer = None  # DataNormalizer(config, db_connection) if db_connection else None
        # TODO: Fix DataNormalizer abstract method implementation
        self.normalizer = None
        
        # Analysis parameters
        self.normalize_data = config.get('spatial_analysis', {}).get('normalize_data', True)
        self.save_results_enabled = config.get('spatial_analysis', {}).get('save_results', True)
        self.output_dir = Path(config.get('spatial_analysis', {}).get('output_dir', 'outputs/spatial_analysis'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self._current_step = 0
        self._total_steps = 0
        
    @abstractmethod
    def analyze(self, 
                data: Union[xr.Dataset, xr.DataArray, np.ndarray],
                **kwargs) -> AnalysisResult:
        """
        Perform spatial analysis on input data.
        
        Args:
            data: Input data in various formats
            **kwargs: Method-specific parameters
            
        Returns:
            AnalysisResult containing labels and metadata
        """
        pass
    
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for this analysis method."""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate analysis parameters.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        pass
    
    def prepare_data(self, 
                    data: Union[xr.Dataset, xr.DataArray, np.ndarray],
                    normalize: Optional[bool] = None,
                    flatten: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Prepare data for analysis.
        
        Args:
            data: Input data in various formats
            normalize: Whether to normalize (uses config default if None)
            flatten: Whether to flatten spatial dimensions
            
        Returns:
            Tuple of (prepared_array, metadata_dict)
        """
        logger.info(f"Preparing data for {self.__class__.__name__}")
        
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
        
        # Normalize if requested
        if normalize is None:
            normalize = self.normalize_data
            
        if normalize and self.normalizer:
            logger.info("Normalizing data")
            norm_result = self.normalizer.normalize(data, save_params=True)
            data = norm_result['data']
            metadata['normalization_params'] = norm_result['parameters']
            metadata['normalization_id'] = norm_result.get('parameter_id')
        
        # Convert to numpy for analysis
        if flatten:
            # Flatten spatial dimensions while preserving band structure
            # Only call xarray_to_numpy for xarray objects, not numpy arrays
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
        metadata['normalized'] = normalize
        
        return prepared_array, metadata
    
    def restore_spatial_structure(self,
                                labels: np.ndarray,
                                metadata: Dict[str, Any]) -> xr.DataArray:
        """
        Restore spatial structure to analysis results.
        
        Args:
            labels: Flattened array of cluster/region labels
            metadata: Metadata from prepare_data
            
        Returns:
            DataArray with spatial structure restored
        """
        logger.info("Restoring spatial structure to results")
        
        if 'coords_info' in metadata:
            # Handle xarray_to_numpy results - need special handling for Dataset inputs
            if 'dims' in metadata and 'shape' in metadata:
                dims = metadata['dims']
                shape = metadata['shape']
                
                if len(dims) == 3 and dims[0] == 'variable':
                    # Dataset was converted to (variable, lat, lon), so spatial shape is shape[1:]
                    spatial_shape = shape[1:]
                    spatial_dims = dims[1:]
                    
                    # Extract coordinate info for spatial dimensions
                    coords = {}
                    coords_info = metadata['coords_info']
                    for dim in spatial_dims:
                        if dim in coords_info:
                            coords[dim] = coords_info[dim]['values']
                    
                    # Reshape to spatial dimensions
                    reshaped = labels.reshape(spatial_shape)
                    
                    # Create DataArray
                    result_da = xr.DataArray(
                        reshaped,
                        coords=coords,
                        dims=spatial_dims,
                        name='analysis_result'
                    )
                else:
                    # DataArray case - use unflatten_spatial
                    result_da = self.array_converter.unflatten_spatial(labels, metadata)
            else:
                # Fallback to unflatten_spatial
                result_da = self.array_converter.unflatten_spatial(labels, metadata)
        else:
            # Manual restoration
            original_shape = metadata.get('original_shape', labels.shape)
            if len(original_shape) == 3:
                # Remove band dimension
                spatial_shape = original_shape[1:]
            else:
                spatial_shape = original_shape
                
            reshaped = labels.reshape(spatial_shape)
            
            # Create DataArray with coordinates if available
            if metadata.get('coords'):
                coords = {}
                for dim, values in metadata['coords'].items():
                    if dim in ['lat', 'lon', 'x', 'y']:
                        coords[dim] = values
                        
                result_da = xr.DataArray(
                    reshaped,
                    coords=coords,
                    dims=list(coords.keys()),
                    name='analysis_result'
                )
            else:
                result_da = xr.DataArray(reshaped, name='analysis_result')
        
        # Add attributes
        result_da.attrs.update({
            'analysis_type': self.__class__.__name__,
            'timestamp': datetime.now().isoformat(),
            'normalized': metadata.get('normalized', False)
        })
        
        return result_da
    
    def save_results(self, result: AnalysisResult, name: Optional[str] = None) -> Path:
        """
        Save analysis results to disk.
        
        Args:
            result: Analysis results
            name: Optional name for output files
            
        Returns:
            Path to saved results
        """
        if name is None:
            name = f"{result.metadata.analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_subdir = self.output_dir / name
        output_subdir.mkdir(exist_ok=True)
        
        # Save labels
        np.save(output_subdir / 'labels.npy', result.labels)
        
        # Save metadata
        with open(output_subdir / 'metadata.json', 'w') as f:
            json.dump(asdict(result.metadata), f, indent=2)
        
        # Save statistics
        with open(output_subdir / 'statistics.json', 'w') as f:
            json.dump(result.statistics, f, indent=2, default=str)
        
        # Save spatial output if present
        if result.spatial_output is not None:
            if isinstance(result.spatial_output, xr.Dataset):
                result.spatial_output.to_netcdf(output_subdir / 'spatial_output.nc')
            else:
                result.spatial_output.to_netcdf(output_subdir / 'spatial_output.nc')
        
        # Save additional outputs
        if result.additional_outputs:
            additional_dir = output_subdir / 'additional'
            additional_dir.mkdir(exist_ok=True)
            
            for key, value in result.additional_outputs.items():
                if isinstance(value, np.ndarray):
                    np.save(additional_dir / f'{key}.npy', value)
                elif isinstance(value, (xr.Dataset, xr.DataArray)):
                    value.to_netcdf(additional_dir / f'{key}.nc')
                elif isinstance(value, gpd.GeoDataFrame):
                    value.to_file(additional_dir / f'{key}.gpkg', driver='GPKG')
                else:
                    # Try to save as JSON
                    try:
                        with open(additional_dir / f'{key}.json', 'w') as f:
                            json.dump(value, f, indent=2, default=str)
                    except:
                        logger.warning(f"Could not save additional output '{key}'")
        
        logger.info(f"Results saved to {output_subdir}")
        return output_subdir
    
    def load_results(self, path: Path) -> AnalysisResult:
        """
        Load previously saved results.
        
        Args:
            path: Path to results directory
            
        Returns:
            Loaded AnalysisResult
        """
        # Load labels
        labels = np.load(path / 'labels.npy')
        
        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata_dict = json.load(f)
            metadata = AnalysisMetadata(**metadata_dict)
        
        # Load statistics
        with open(path / 'statistics.json', 'r') as f:
            statistics = json.load(f)
        
        # Load spatial output if exists
        spatial_output = None
        spatial_path = path / 'spatial_output.nc'
        if spatial_path.exists():
            spatial_output = xr.open_dataset(spatial_path)
            # Convert to DataArray if single variable
            if len(spatial_output.data_vars) == 1:
                spatial_output = spatial_output[list(spatial_output.data_vars)[0]]
        
        # Load additional outputs
        additional_outputs = {}
        additional_dir = path / 'additional'
        if additional_dir.exists():
            for file_path in additional_dir.iterdir():
                key = file_path.stem
                if file_path.suffix == '.npy':
                    additional_outputs[key] = np.load(file_path)
                elif file_path.suffix == '.nc':
                    additional_outputs[key] = xr.open_dataset(file_path)
                elif file_path.suffix == '.gpkg':
                    additional_outputs[key] = gpd.read_file(file_path)
                elif file_path.suffix == '.json':
                    with open(file_path, 'r') as f:
                        additional_outputs[key] = json.load(f)
        
        return AnalysisResult(
            labels=labels,
            metadata=metadata,
            statistics=statistics,
            spatial_output=spatial_output,
            additional_outputs=additional_outputs if additional_outputs else None
        )
    
    def store_in_database(self, result: AnalysisResult) -> Optional[int]:
        """
        Store analysis results in database.
        
        Args:
            result: Analysis results to store
            
        Returns:
            Database record ID if successful
        """
        if not self.db or not self.save_results_enabled:
            return None
        
        try:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                
                # Store main result record
                cur.execute("""
                    INSERT INTO spatial_analysis_results
                    (analysis_type, input_shape, parameters, statistics, 
                     processing_time, created_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    result.metadata.analysis_type,
                    list(result.metadata.input_shape),
                    json.dumps(result.metadata.parameters),
                    json.dumps(result.statistics),
                    result.metadata.processing_time,
                    datetime.now(),
                    json.dumps(asdict(result.metadata))
                ))
                
                result_id = cur.fetchone()[0]
                
                # Store labels summary (not full array)
                unique_labels, counts = np.unique(result.labels, return_counts=True)
                cur.execute("""
                    INSERT INTO spatial_analysis_labels
                    (result_id, unique_labels, label_counts, total_pixels)
                    VALUES (%s, %s, %s, %s)
                """, (
                    result_id,
                    unique_labels.tolist(),
                    counts.tolist(),
                    len(result.labels)
                ))
                
                conn.commit()
                logger.info(f"Stored analysis results with ID: {result_id}")
                return result_id
                
        except Exception as e:
            logger.error(f"Failed to store results in database: {e}")
            return None
    
    def _update_progress(self, step: int, total: int, message: str = ""):
        """Update progress tracking."""
        self._current_step = step
        self._total_steps = total
        
        progress = (step / total) * 100 if total > 0 else 0
        logger.info(f"Progress: {progress:.1f}% - {message}")
        
        # Call parent progress callback if set
        if self._progress_callback:
            self._progress_callback(progress)
    
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
                issues.append("Empty array provided")
        elif isinstance(data, (xr.Dataset, xr.DataArray)):
            if isinstance(data, xr.Dataset) and len(data.data_vars) == 0:
                issues.append("Dataset has no variables")
            elif isinstance(data, xr.Dataset):
                # Check if Dataset is empty by checking sizes
                if all(data[var].size == 0 for var in data.data_vars):
                    issues.append("Empty data provided")
            elif data.size == 0:
                issues.append("Empty data provided")
        
        # Check for all NaN/null values
        if isinstance(data, np.ndarray):
            if np.all(np.isnan(data)):
                issues.append("All values are NaN")
        else:
            # xarray
            if isinstance(data, xr.Dataset):
                for var in data.data_vars:
                    if data[var].isnull().all():
                        issues.append(f"Variable '{var}' contains all null values")
            else:
                if data.isnull().all():
                    issues.append("All values are null")
        
        # Check dimensions
        if isinstance(data, np.ndarray):
            if data.ndim not in [2, 3]:
                issues.append(f"Array must be 2D or 3D, got {data.ndim}D")
        
        return len(issues) == 0, issues
    # Required BaseProcessor abstract methods
    def process_single(self, item: Any) -> Any:
        """Process a single item - default implementation for spatial analysis."""
        # For spatial analyzers, this is typically not used directly
        # as they work with entire datasets
        return self.analyze(item)
    
    def validate_input(self, item: Any) -> Tuple[bool, Optional[str]]:
        """Validate input item."""
        is_valid, issues = self.validate_input_data(item)
        error_msg = "; ".join(issues) if issues else None
        return is_valid, error_msg