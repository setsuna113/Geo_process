# src/base/analyzer.py
"""Base analyzer implementation with common functionality for all analysis types."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, List, Callable
import logging
import numpy as np
import xarray as xr
from datetime import datetime
import time

# Import interface and types from abstractions
from src.abstractions.interfaces.analyzer import (
    IAnalyzer, AnalysisResult, AnalysisMetadata
)
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseAnalyzer(IAnalyzer, ABC):
    """
    Base analyzer with common implementation for all analysis types.
    
    This class provides:
    - Configuration management (handles both dict and Config objects)
    - Common data preparation methods
    - Result storage coordination
    - Progress tracking capabilities
    - Memory-aware processing
    - Component initialization
    
    Subclasses should implement the abstract methods for specific analysis types.
    """
    
    def __init__(self, 
                 config: Union[Dict[str, Any], Any], 
                 db_connection: Optional[Any] = None,
                 **kwargs):
        """
        Initialize base analyzer with flexible configuration.
        
        Args:
            config: Configuration dict or Config object
            db_connection: Optional database connection
            **kwargs: Additional parameters for subclasses
        """
        # Handle both dict and Config object patterns safely
        if hasattr(config, 'config') and isinstance(config.config, dict):
            # This is a Config object with internal dict
            self.config = config.config
            self.config_obj = config
        elif isinstance(config, dict):
            # This is already a dict
            self.config = config
            self.config_obj = None
        else:
            # Assume it's a Config-like object
            self.config = config
            self.config_obj = config
            
        self.db = db_connection
        
        # Set analysis type from class name
        self.analysis_type = self.__class__.__name__.replace('Analyzer', '').upper()
        
        # Optional components (lazy initialized)
        self._array_converter = None
        self._normalizer = None
        
        # Progress callback
        self._progress_callback = None
        
        # Configuration for saving results
        self.save_results_enabled = self.safe_get_config('analysis.save_results.enabled', True)
        
        # Initialize any additional components from kwargs
        self._init_components(**kwargs)
        
        logger.info(f"Initialized {self.__class__.__name__} with analysis type: {self.analysis_type}")
    
    def _init_components(self, **kwargs):
        """
        Initialize optional components. Override in subclasses for specific needs.
        
        Args:
            **kwargs: Component-specific parameters
        """
        # Subclasses can override to initialize specific components
        pass
    
    def prepare_data(self, 
                    data: Union[xr.Dataset, xr.DataArray, np.ndarray],
                    normalize: bool = True,
                    handle_nan: bool = True) -> np.ndarray:
        """
        Common data preparation logic.
        
        Args:
            data: Input data in various formats
            normalize: Whether to normalize the data
            handle_nan: Whether to handle NaN values
            
        Returns:
            Prepared numpy array
        """
        # Convert to numpy array if needed
        if isinstance(data, xr.Dataset):
            # If Dataset, stack all data variables
            arrays = []
            for var in data.data_vars:
                arrays.append(data[var].values)
            data = np.stack(arrays, axis=-1)
        elif isinstance(data, xr.DataArray):
            data = data.values
        elif not isinstance(data, np.ndarray):
            data = np.array(data)
            
        # Handle NaN values
        if handle_nan and np.any(np.isnan(data)):
            logger.warning(f"Found {np.sum(np.isnan(data))} NaN values, replacing with 0")
            data = np.nan_to_num(data, nan=0.0)
            
        # Normalize if requested
        if normalize and self._normalizer is not None:
            try:
                data = self._normalizer.normalize(data)
            except Exception as e:
                logger.warning(f"Normalization failed: {e}, continuing with unnormalized data")
        
        return data
    
    def create_metadata(self, 
                       data_shape: Tuple[int, ...],
                       parameters: Dict[str, Any],
                       processing_time: float,
                       **extra_fields) -> AnalysisMetadata:
        """
        Create standardized metadata for analysis results.
        
        Args:
            data_shape: Shape of input data
            parameters: Analysis parameters used
            processing_time: Time taken for analysis
            **extra_fields: Additional metadata fields
            
        Returns:
            AnalysisMetadata instance
        """
        metadata = AnalysisMetadata(
            analysis_type=self.analysis_type,
            input_shape=data_shape,
            input_bands=parameters.get('bands', []),
            parameters=parameters.copy(),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            data_source=extra_fields.get('data_source'),
            normalization_applied=extra_fields.get('normalization_applied', False),
            coordinate_system=extra_fields.get('coordinate_system', 'EPSG:4326')
        )
        
        return metadata
    
    def validate_input_data(self, data: Union[xr.Dataset, xr.DataArray, np.ndarray]) -> Tuple[bool, List[str]]:
        """
        Common validation logic for input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for None
        if data is None:
            issues.append("Data is None")
            return False, issues
            
        # Check shape
        if hasattr(data, 'shape'):
            if len(data.shape) == 0:
                issues.append("Data has no dimensions")
            elif any(dim == 0 for dim in data.shape):
                issues.append(f"Data has zero-length dimension: shape={data.shape}")
        else:
            issues.append("Data has no shape attribute")
            
        # Check for all NaN
        if isinstance(data, (np.ndarray, xr.DataArray, xr.Dataset)):
            if isinstance(data, (xr.DataArray, xr.Dataset)):
                values = data.values if hasattr(data, 'values') else data
            else:
                values = data
                
            if np.all(np.isnan(values)):
                issues.append("All values are NaN")
                
        # Check data type
        if hasattr(data, 'dtype'):
            if not np.issubdtype(data.dtype, np.number):
                issues.append(f"Data type {data.dtype} is not numeric")
                
        return len(issues) == 0, issues
    
    def safe_get_config(self, key: str, default: Any = None) -> Any:
        """
        Safely get configuration value, handling various config formats.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            # Handle dot notation
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k, default)
                elif hasattr(value, k):
                    value = getattr(value, k)
                else:
                    return default
                    
            return value if value is not None else default
            
        except Exception:
            return default
    
    def _update_progress(self, current: int, total: int, message: str):
        """
        Update progress through callback if set.
        
        Args:
            current: Current step number
            total: Total number of steps
            message: Progress message
        """
        if self._progress_callback:
            try:
                self._progress_callback(current, total, message)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """
        Set callback for progress updates.
        
        Args:
            callback: Function that accepts (current, total, message)
        """
        self._progress_callback = callback
    
    def save_results(self, result: AnalysisResult, output_name: str, 
                     output_dir: Path = None) -> Path:
        """
        Default implementation for saving results.
        
        Args:
            result: Analysis result to save
            output_name: Base name for output files
            output_dir: Directory to save results (optional)
            
        Returns:
            Path to the primary saved file
        """
        import pickle
        import json
        
        # Use default output directory if not provided
        if output_dir is None:
            output_dir = Path(self.safe_get_config('analysis.save_results.output_dir', 'outputs/analysis'))
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get configured formats
        formats = self.safe_get_config('analysis.save_results.formats', ['pkl'])
        saved_paths = []
        
        # Save in each configured format
        if 'pkl' in formats:
            pkl_path = output_dir / f"{output_name}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(result, f)
            saved_paths.append(pkl_path)
            logger.info(f"Saved pickle format to: {pkl_path}")
        
        if 'json' in formats:
            json_path = output_dir / f"{output_name}.json"
            # Convert to JSON-serializable format
            json_data = {
                'metadata': result.metadata.__dict__,
                'statistics': result.statistics,
                'labels_shape': result.labels.shape,
                'labels_dtype': str(result.labels.dtype)
            }
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            saved_paths.append(json_path)
            logger.info(f"Saved JSON metadata to: {json_path}")
        
        if 'npy' in formats:
            npy_path = output_dir / f"{output_name}_labels.npy"
            np.save(npy_path, result.labels)
            saved_paths.append(npy_path)
            logger.info(f"Saved labels array to: {npy_path}")
        
        # Return primary saved path (first format)
        return saved_paths[0] if saved_paths else None
    
    # Default implementations for IAnalyzer methods
    # Subclasses should override these
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default parameters. Subclasses should override.
        
        Returns:
            Empty dict by default
        """
        return {}
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameters. Subclasses should override for specific validation.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            (True, []) by default - subclasses should implement actual validation
        """
        return True, []
    
    @abstractmethod
    def analyze(self, 
                data: Union[xr.Dataset, xr.DataArray, np.ndarray],
                **kwargs) -> AnalysisResult:
        """
        Perform the analysis. Must be implemented by subclasses.
        
        Args:
            data: Input data
            **kwargs: Analysis-specific parameters
            
        Returns:
            AnalysisResult with analysis outputs
        """
        pass