# src/spatial_analysis/base_analyzer.py
"""Ultimate consolidated base class for spatial analysis methods."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import xarray as xr
import geopandas as gpd
from dataclasses import dataclass, asdict
import psutil

# Import components for backward compatibility with tests
try:
    from src.processors.data_preparation.array_converter import ArrayConverter
except ImportError:
    ArrayConverter = None

try:
    from src.processors.data_preparation.data_normalizer import DataNormalizer
except ImportError:
    DataNormalizer = None

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

class BaseAnalyzer(ABC):
    """
    Ultimate consolidated base class for spatial analysis methods.
    
    Combines the best features from all previous versions:
    - Pure abstract interface from base_analyzer_new.py
    - Comprehensive functionality from base_analyzer.py
    - Factory pattern for analyzer creation
    - Proper dependency injection support
    - Backward compatibility with existing code
    """
    
    def __init__(self, config: Union[Dict[str, Any], Any], 
                 db_connection: Optional[Any] = None,
                 data_processor: Optional[Any] = None,
                 result_store: Optional[Any] = None,
                 analysis_store: Optional[Any] = None,
                 **kwargs):
        """
        Initialize analyzer with flexible dependency injection.
        
        Args:
            config: Configuration (dict or Config object)
            db_connection: Optional database connection
            data_processor: Optional data processor instance
            result_store: Optional file-based result store
            analysis_store: Optional database result store
            **kwargs: Additional initialization parameters
        """
        # Handle both dict and Config object
        if hasattr(config, 'config'):
            self.config = config.config
            self.config_obj = config
        elif isinstance(config, dict):
            self.config = config
            self.config_obj = None
        else:
            self.config = config
            self.config_obj = config
        
        # Store dependencies (can be None for minimal usage)
        self.db = db_connection
        self.data_processor = data_processor
        self.result_store = result_store
        self.analysis_store = analysis_store
        
        # Initialize components (eager loading for backward compatibility)
        self._array_converter = None
        self._normalizer = None
        
        # Initialize components if available (for backward compatibility with tests)
        if not self.data_processor:
            try:
                if ArrayConverter and (self.config_obj or isinstance(self.config, dict)):
                    self._array_converter = ArrayConverter(self.config_obj or self.config)
            except Exception as e:
                logger.debug(f"Could not initialize ArrayConverter: {e}")
            
            try:
                if DataNormalizer and self.db and (self.config_obj or isinstance(self.config, dict)):
                    self._normalizer = DataNormalizer(self.config_obj or self.config, self.db)
            except Exception as e:
                logger.debug(f"Could not initialize DataNormalizer: {e}")
        
        # Analysis metadata
        self.analysis_type = self.__class__.__name__.replace('Analyzer', '').upper()
        
        # Configuration-driven settings (simple and robust)
        def safe_get_config(config_dict, key, default):
            """Safely get config value, handling both dicts and Mock objects."""
            try:
                if hasattr(config_dict, 'get') and callable(config_dict.get):
                    return config_dict.get(key, default)
                elif hasattr(config_dict, key):
                    return getattr(config_dict, key)
                else:
                    return default
            except:
                return default
        
        # Get spatial analysis config section
        spatial_config = safe_get_config(self.config, 'spatial_analysis', {})
        
        # Extract individual settings
        self.normalize_data = safe_get_config(spatial_config, 'normalize_data', True)
        self.save_results_enabled = safe_get_config(spatial_config, 'save_results', True)
        output_dir_value = safe_get_config(spatial_config, 'output_dir', 'outputs/spatial_analysis')
        
        # Handle output_dir safely for mocked configs
        try:
            self.output_dir = Path(output_dir_value)
            if self.save_results_enabled:
                self.output_dir.mkdir(parents=True, exist_ok=True)
        except (TypeError, OSError) as e:
            # Fallback for tests or invalid paths
            logger.debug(f"Could not create output directory: {e}")
            self.output_dir = Path('outputs/spatial_analysis')
        
        # Progress tracking
        self._current_step = 0
        self._total_steps = 0
    
    @property
    def array_converter(self):
        """Array converter (lazy-loaded or injected)."""
        if self._array_converter is None and self.data_processor:
            self._array_converter = self.data_processor.array_converter
        return self._array_converter
    
    @property
    def normalizer(self):
        """Lazy-loaded normalizer."""
        if self._normalizer is None:
            # Try to initialize normalizer if components are available
            try:
                from src.processors.data_preparation.data_normalizer import DataNormalizer
                if self.db:
                    self._normalizer = DataNormalizer(self.config_obj or self.config, self.db)
            except ImportError:
                logger.debug("DataNormalizer not available")
        return self._normalizer
    
    # === ABSTRACT INTERFACE (must be implemented by subclasses) ===
    
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default parameters for this analysis method.
        
        Returns:
            Dictionary of default parameter values
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate analysis parameters.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        pass
    
    @abstractmethod
    def analyze(self, 
                data: Union[xr.Dataset, xr.DataArray, np.ndarray],
                **kwargs) -> AnalysisResult:
        """
        Perform the spatial analysis.
        
        Args:
            data: Input data for analysis
            **kwargs: Analysis-specific parameters
            
        Returns:
            AnalysisResult containing labels, metadata, and statistics
        """
        pass
    
    # === CONCRETE FUNCTIONALITY (shared across all analyzers) ===
    
    def prepare_data(self, 
                    data: Union[xr.Dataset, xr.DataArray, np.ndarray],
                    normalize: Optional[bool] = None,
                    flatten: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Prepare data for analysis.
        
        Uses injected data_processor if available, otherwise fallback implementation.
        """
        if self.data_processor:
            return self.data_processor.prepare_data(data, normalize, flatten)
        else:
            # Fallback implementation for backward compatibility
            return self._fallback_prepare_data(data, normalize, flatten)
    
    def _fallback_prepare_data(self, data, normalize, flatten):
        """Fallback data preparation when no data_processor is injected."""
        logger.info(f"Preparing data for {self.__class__.__name__} (fallback mode)")
        
        # Convert to xarray if needed
        if isinstance(data, np.ndarray):
            if data.ndim == 3:
                data = xr.DataArray(
                    data,
                    dims=['band', 'y', 'x'],
                    coords={'band': [f'band_{i}' for i in range(data.shape[0])]}
                )
            else:
                raise ValueError("Numpy array must be 3D (bands, height, width)")
        
        # Store metadata
        if isinstance(data, xr.Dataset):
            first_var = list(data.data_vars)[0]
            spatial_shape = data[first_var].shape
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
            if isinstance(data, (xr.Dataset, xr.DataArray)):
                if self.array_converter:
                    result = self.array_converter.xarray_to_numpy(data, flatten=True, preserve_coords=True)
                    prepared_array = result['array']
                    metadata.update(result)
                else:
                    # Simple fallback
                    if isinstance(data, xr.Dataset):
                        prepared_array = np.stack([data[var].values for var in data.data_vars]).flatten()
                    else:
                        prepared_array = data.values.flatten()
            else:
                prepared_array = data.flatten()
                metadata['coords_info'] = None
            
            # Reshape to (n_samples, n_features) for sklearn-like interface
            if prepared_array.ndim == 1:
                prepared_array = prepared_array.reshape(-1, 1)
        else:
            # Keep spatial structure
            if isinstance(data, xr.Dataset):
                prepared_array = np.stack([data[var].values for var in data.data_vars])
            elif isinstance(data, xr.DataArray):
                prepared_array = data.values
            else:
                prepared_array = data
            
            metadata['spatial_shape'] = prepared_array.shape
        
        metadata['prepared_shape'] = prepared_array.shape
        metadata['normalized'] = normalize and self.normalizer is not None
        
        return prepared_array, metadata
    
    def restore_spatial_structure(self, labels: np.ndarray, 
                                 metadata: Dict[str, Any]) -> Optional[Union[xr.Dataset, xr.DataArray]]:
        """Restore spatial structure to flat analysis results."""
        if self.data_processor:
            return self.data_processor.restore_spatial_structure(labels, metadata)
        elif self.array_converter and 'coords_info' in metadata:
            try:
                spatial_output = self.array_converter.numpy_to_xarray(
                    labels, 
                    metadata['coords_info'], 
                    metadata.get('dims', ['y', 'x'])
                )
                
                if isinstance(spatial_output, xr.DataArray):
                    spatial_output.name = 'cluster_labels'
                    spatial_output.attrs['long_name'] = 'Spatial cluster assignments'
                
                return spatial_output
            except Exception as e:
                logger.warning(f"Could not restore spatial structure: {e}")
        
        return None
    
    def save_results(self, result: AnalysisResult, name: Optional[str] = None) -> Path:
        """Save analysis results to files."""
        if self.result_store:
            experiment_name = name or f"{self.analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            return self.result_store.save_results(result, experiment_name)
        else:
            # Fallback implementation
            return self._fallback_save_results(result, name)
    
    def _fallback_save_results(self, result: AnalysisResult, name: Optional[str] = None) -> Path:
        """Fallback save implementation."""
        if not name:
            name = f"{self.analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_subdir = self.output_dir / name
        output_subdir.mkdir(exist_ok=True)
        
        # Save labels
        np.save(output_subdir / 'labels.npy', result.labels)
        
        # Save metadata
        metadata_dict = {
            'analysis_type': result.metadata.analysis_type,
            'input_shape': result.metadata.input_shape,
            'input_bands': result.metadata.input_bands,
            'parameters': result.metadata.parameters,
            'processing_time': result.metadata.processing_time,
            'timestamp': result.metadata.timestamp,
            'data_source': result.metadata.data_source,
            'normalization_applied': result.metadata.normalization_applied,
            'coordinate_system': result.metadata.coordinate_system
        }
        
        with open(output_subdir / 'metadata.json', 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        
        # Save statistics
        if result.statistics:
            with open(output_subdir / 'statistics.json', 'w') as f:
                json.dump(result.statistics, f, indent=2, default=str)
        
        # Save spatial output if available
        if result.spatial_output is not None:
            if isinstance(result.spatial_output, (xr.Dataset, xr.DataArray)):
                result.spatial_output.to_netcdf(output_subdir / 'spatial_output.nc')
            elif isinstance(result.spatial_output, gpd.GeoDataFrame):
                result.spatial_output.to_file(output_subdir / 'spatial_output.gpkg', driver='GPKG')
        
        # Save additional outputs
        if result.additional_outputs:
            additional_dir = output_subdir / 'additional_outputs'
            additional_dir.mkdir(exist_ok=True)
            
            for key, value in result.additional_outputs.items():
                try:
                    if isinstance(value, np.ndarray):
                        np.save(additional_dir / f'{key}.npy', value)
                    elif isinstance(value, (xr.Dataset, xr.DataArray)):
                        value.to_netcdf(additional_dir / f'{key}.nc')
                    elif isinstance(value, gpd.GeoDataFrame):
                        value.to_file(additional_dir / f'{key}.gpkg', driver='GPKG')
                    else:
                        with open(additional_dir / f'{key}.json', 'w') as f:
                            json.dump(value, f, indent=2, default=str)
                except Exception as e:
                    logger.warning(f"Could not save additional output '{key}': {e}")
        
        logger.info(f"Results saved to {output_subdir}")
        return output_subdir
    
    def store_in_database(self, result: AnalysisResult, experiment_name: Optional[str] = None) -> Optional[int]:
        """Store results in database."""
        if self.analysis_store and experiment_name:
            try:
                return self.analysis_store.store_result(result, experiment_name)
            except Exception as e:
                logger.error(f"Failed to store in database: {e}")
        elif self.db and experiment_name:
            # Fallback database storage
            try:
                return self._fallback_store_in_database(result, experiment_name)
            except Exception as e:
                logger.error(f"Failed to store in database (fallback): {e}")
        
        return None
    
    def _fallback_store_in_database(self, result: AnalysisResult, experiment_name: str) -> Optional[int]:
        """Fallback database storage."""
        # This would implement basic database storage without AnalysisStore
        # For now, just log that it would be stored
        logger.info(f"Would store {experiment_name} in database (fallback)")
        return None
    
    def load_results(self, path_or_name: Union[Path, str]) -> Optional[AnalysisResult]:
        """Load previously saved results."""
        if self.result_store and isinstance(path_or_name, str):
            return self.result_store.load_results(path_or_name)
        else:
            # Fallback implementation for path-based loading
            return self._fallback_load_results(Path(path_or_name))
    
    def _fallback_load_results(self, path: Path) -> Optional[AnalysisResult]:
        """Fallback load implementation."""
        if not path.exists():
            return None
        
        try:
            # Load basic components
            labels = np.load(path / 'labels.npy')
            
            with open(path / 'metadata.json', 'r') as f:
                metadata_dict = json.load(f)
            
            metadata = AnalysisMetadata(
                analysis_type=metadata_dict['analysis_type'],
                input_shape=tuple(metadata_dict['input_shape']),
                input_bands=metadata_dict['input_bands'],
                parameters=metadata_dict['parameters'],
                processing_time=metadata_dict['processing_time'],
                timestamp=metadata_dict['timestamp'],
                data_source=metadata_dict.get('data_source'),
                normalization_applied=metadata_dict.get('normalization_applied', False),
                coordinate_system=metadata_dict.get('coordinate_system', 'EPSG:4326')
            )
            
            statistics = {}
            statistics_path = path / 'statistics.json'
            if statistics_path.exists():
                with open(statistics_path, 'r') as f:
                    statistics = json.load(f)
            
            return AnalysisResult(
                labels=labels,
                metadata=metadata,
                statistics=statistics,
                spatial_output=None,  # Would need more complex loading
                additional_outputs=None
            )
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return None
    
    def validate_input_data(self, data: Union[xr.Dataset, xr.DataArray, np.ndarray]) -> Tuple[bool, List[str]]:
        """Validate input data."""
        if self.data_processor:
            return self.data_processor.validate_input_data(data)
        else:
            # Fallback validation
            issues = []
            
            if not isinstance(data, (xr.Dataset, xr.DataArray, np.ndarray)):
                issues.append(f"Unsupported data type: {type(data)}")
                return False, issues
            
            if isinstance(data, np.ndarray):
                if data.size == 0:
                    issues.append("Data array is empty")
                if np.all(np.isnan(data)):
                    issues.append("All data values are NaN")
                if data.ndim < 2:
                    issues.append("Data must be at least 2-dimensional")
            
            return len(issues) == 0, issues
    
    def estimate_memory_requirements(self, data_shape: Tuple[int, ...], 
                                   dtype: np.dtype = np.dtype(np.float64)) -> Dict[str, Any]:
        """Estimate memory requirements for analysis."""
        if self.data_processor:
            return self.data_processor.estimate_memory_requirements(data_shape, dtype)
        else:
            # Fallback estimation
            element_size = dtype.itemsize
            n_elements = np.prod(data_shape)
            data_size_gb = (n_elements * element_size) / (1024**3)
            
            overhead_factor = getattr(self, 'memory_overhead_factor', 2.0)
            total_required_gb = data_size_gb * overhead_factor
            
            # Get system memory
            memory = psutil.virtual_memory()
            
            return {
                'data_size_gb': float(data_size_gb),
                'overhead_factor': float(overhead_factor),
                'total_required_gb': float(total_required_gb),
                'available_gb': float(memory.available / (1024**3)),
                'total_system_gb': float(memory.total / (1024**3)),
                'fits_in_memory': bool(total_required_gb < memory.available * 0.8),
                'estimated_by': 'base_analyzer'
            }
    
    # === PROGRESS TRACKING ===
    
    def update_progress(self, current_step: int, total_steps: int, description: str = ""):
        """Update progress tracking."""
        self._current_step = current_step
        self._total_steps = total_steps
        
        if total_steps > 0:
            progress_pct = (current_step / total_steps) * 100
            logger.info(f"Progress: {progress_pct:.1f}% - {description}")
    
    def _update_progress(self, step: int, total: int, message: str = ""):
        """Legacy progress update method for backward compatibility."""
        self.update_progress(step, total, message)
    
    def get_progress(self) -> Tuple[int, int, float]:
        """Get current progress."""
        progress_pct = (self._current_step / self._total_steps * 100) if self._total_steps > 0 else 0
        return self._current_step, self._total_steps, progress_pct
    
    # === DATA GENERATION (for large datasets) ===
    
    def create_data_generator(self, data_path: str, chunk_size: int = 50000):
        """Create a data generator for processing large datasets in chunks."""
        # This would be implemented based on the specific data format
        # For now, it's a placeholder that could be overridden by subclasses
        logger.warning("create_data_generator not implemented in base class")
        return None

# === FACTORY PATTERN ===

class AnalysisFactory:
    """Factory for creating analyzer instances with proper dependency injection."""
    
    _registry = {}
    
    @classmethod
    def register(cls, analysis_type: str, analyzer_class: type):
        """Register an analyzer class."""
        cls._registry[analysis_type.lower()] = analyzer_class
    
    @classmethod
    def create(cls, analysis_type: str, config: Union[Dict[str, Any], Any], 
               **dependencies) -> BaseAnalyzer:
        """
        Create an analyzer instance with dependency injection.
        
        Args:
            analysis_type: Type of analyzer to create
            config: Configuration
            **dependencies: Components to inject (db_connection, data_processor, etc.)
        """
        analyzer_class = cls._registry.get(analysis_type.lower())
        if not analyzer_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown analysis type '{analysis_type}'. Available: {available}")
        
        return analyzer_class(config, **dependencies)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available analysis types."""
        return list(cls._registry.keys())

# === BACKWARD COMPATIBILITY ===

# Support both old BaseProcessor-style usage and new dependency injection
def create_analyzer_with_backward_compatibility(analyzer_class, config, db_connection=None, **kwargs):
    """Helper function to create analyzers with backward compatibility."""
    try:
        # Try new dependency injection style first
        return analyzer_class(config, db_connection=db_connection, **kwargs)
    except TypeError:
        # Fallback to old style if needed
        if hasattr(config, 'config'):
            return analyzer_class(config, db_connection, **kwargs)
        else:
            return analyzer_class(config, **kwargs)