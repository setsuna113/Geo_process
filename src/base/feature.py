"""Base feature extractor class."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging

from ..core.registry import component_registry
from src.config import config
from ..database.schema import schema

logger = logging.getLogger(__name__)

class SourceType(Enum):
    """Types of data sources for feature extraction."""
    RASTER = "raster"
    VECTOR = "vector"
    TABULAR = "tabular"
    API = "api"
    COMPUTED = "computed"
    MIXED = "mixed"

@dataclass
class FeatureResult:
    """Standard feature extraction result."""
    feature_name: str
    feature_type: str
    value: Union[float, np.ndarray]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        value = self.value
        if isinstance(value, np.ndarray):
            value = value.tolist()
            
        return {
            'feature_name': self.feature_name,
            'feature_type': self.feature_type,
            'feature_value': value,
            'computation_metadata': self.metadata or {}
        }


class BaseFeature(ABC):
    """
    Base class for feature extractors.
    
    Handles:
    - Feature computation
    - Result formatting
    - Storage integration
    """
    
    def __init__(self,
                 feature_type: str,
                 normalize: bool = False,
                 store_results: bool = True,
                 **kwargs):
        """
        Initialize feature extractor.
        
        Args:
            feature_type: Type of feature (e.g., 'species_richness', 'climate')
            normalize: Whether to normalize features
            store_results: Whether to store results in database
            **kwargs: Feature-specific parameters
        """
        self.feature_type = feature_type
        self.normalize = normalize
        self.store_results = store_results
        self.config = self._merge_config(kwargs)
        
    def _merge_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge kwargs with default config."""
        feature_name = self.__class__.__name__
        default_config = config.get(f'features.{feature_name}', {})
        return {**default_config, **kwargs}
    
    @abstractmethod
    def extract_single(self, 
                      grid_cell_id: str,
                      data: Dict[str, Any]) -> List[FeatureResult]:
        """
        Extract features for a single grid cell.
        
        Args:
            grid_cell_id: Grid cell identifier
            data: Input data for the cell
            
        Returns:
            List of FeatureResult objects
        """
        pass
    
    @abstractmethod
    def get_required_data(self) -> List[str]:
        """
        Get list of required data types.
        
        Returns:
            List of data type identifiers
        """
        pass
    
    def validate_data(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate input data.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required = self.get_required_data()
        missing = [r for r in required if r not in data]
        
        if missing:
            return False, f"Missing required data: {missing}"
            
        return True, None
    
    def extract_batch(self,
                     grid_id: str,
                     cell_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[FeatureResult]]:
        """
        Extract features for multiple cells.
        
        Args:
            grid_id: Grid identifier
            cell_data: Dict mapping cell_id to data
            
        Returns:
            Dict mapping cell_id to feature results
        """
        results = {}
        
        for cell_id, data in cell_data.items():
            # Validate data
            is_valid, error = self.validate_data(data)
            if not is_valid:
                logger.warning(f"Invalid data for cell {cell_id}: {error}")
                continue
                
            try:
                # Extract features
                features = self.extract_single(cell_id, data)
                
                # Normalize if requested
                if self.normalize and features:
                    features = self._normalize_features(features)
                    
                results[cell_id] = features
                
                # Store if configured
                if self.store_results and features:
                    self._store_features(grid_id, cell_id, features)
                    
            except Exception as e:
                logger.error(f"Feature extraction failed for cell {cell_id}: {e}")
                
        return results
    
    def _normalize_features(self, features: List[FeatureResult]) -> List[FeatureResult]:
        """Normalize feature values."""
        # Simple min-max normalization
        values = [f.value for f in features if isinstance(f.value, (int, float))]
        
        if not values:
            return features
            
        min_val = min(values)
        max_val = max(values)
        
        if max_val - min_val == 0:
            return features
            
        normalized = []
        for feature in features:
            if isinstance(feature.value, (int, float)):
                norm_value = (feature.value - min_val) / (max_val - min_val)
                normalized.append(FeatureResult(
                    feature_name=feature.feature_name,
                    feature_type=feature.feature_type,
                    value=norm_value,
                    metadata={**(feature.metadata or {}), 'normalized': True}
                ))
            else:
                normalized.append(feature)
                
        return normalized
    
    def _store_features(self, grid_id: str, cell_id: str, features: List[FeatureResult]):
        """Store features in database."""
        feature_dicts = []
        
        for feature in features:
            feature_dict = feature.to_dict()
            feature_dict.update({
                'grid_id': grid_id,
                'cell_id': cell_id
            })
            feature_dicts.append(feature_dict)
            
        if feature_dicts:
            schema.store_features_batch(feature_dicts)
            
    def compute_statistics(self, features: Dict[str, List[FeatureResult]]) -> Dict[str, Any]:
        """Compute statistics across all features."""
        all_values = []
        
        for cell_features in features.values():
            for feature in cell_features:
                if isinstance(feature.value, (int, float)):
                    all_values.append(feature.value)
                    
        if not all_values:
            return {}
            
        return {
            'count': len(all_values),
            'mean': np.mean(all_values),
            'std': np.std(all_values),
            'min': np.min(all_values),
            'max': np.max(all_values),
            'median': np.median(all_values)
        }
    
    # Enhanced methods for base class enhancements
    
    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """
        Get the source type for this feature extractor.
        
        Returns:
            SourceType enum value
        """
        pass
    
    def supports_multi_band(self) -> bool:
        """
        Check if this feature extractor supports multi-band raster data.
        Default implementation returns False - override in subclasses.
        
        Returns:
            True if multi-band data is supported
        """
        return False
        
    @property
    def required_raster_bands(self) -> Optional[List[str]]:
        """
        Get required raster bands for feature extraction.
        Override in subclasses that work with specific bands.
        
        Returns:
            List of required band names or None for any bands
        """
        return None
        
    def validate_raster_bands(self, available_bands: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate that required raster bands are available.
        
        Args:
            available_bands: List of available band names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required = self.required_raster_bands
        if not required:
            return True, None
            
        missing = [band for band in required if band not in available_bands]
        if missing:
            return False, f"Missing required raster bands: {missing}"
            
        return True, None
        
    def extract_from_multi_band(self,
                               grid_cell_id: str,
                               band_data: Dict[str, np.ndarray],
                               metadata: Optional[Dict[str, Any]] = None) -> List[FeatureResult]:
        """
        Extract features from multi-band raster data.
        
        Args:
            grid_cell_id: Grid cell identifier
            band_data: Dictionary mapping band names to data arrays
            metadata: Optional metadata for the data
            
        Returns:
            List of FeatureResult objects
            
        Raises:
            NotImplementedError: If multi-band support not implemented
        """
        if not self.supports_multi_band():
            raise NotImplementedError(f"{self.__class__.__name__} does not support multi-band data")
            
        # Validate bands
        is_valid, error = self.validate_raster_bands(list(band_data.keys()))
        if not is_valid:
            raise ValueError(error)
            
        # Default implementation - override in subclasses
        return []
        
    def estimate_computation_time(self, 
                                data_size_mb: float,
                                cell_count: int) -> Dict[str, float]:
        """
        Estimate computation time for feature extraction.
        
        Args:
            data_size_mb: Size of input data in MB
            cell_count: Number of cells to process
            
        Returns:
            Dictionary with time estimates
        """
        # Base estimates - override in subclasses for more accurate predictions
        base_time_per_cell = 0.01  # 10ms per cell
        data_processing_factor = data_size_mb * 0.001  # 1ms per MB
        
        estimated_seconds = (cell_count * base_time_per_cell) + data_processing_factor
        
        # Add complexity factors based on feature type
        complexity_multipliers = {
            'species_richness': 1.5,
            'climate': 1.0,
            'elevation': 0.5,
            'ndvi': 0.8,
            'texture': 2.0,
            'spectral': 1.2
        }
        
        multiplier = complexity_multipliers.get(self.feature_type, 1.0)
        estimated_seconds *= multiplier
        
        return {
            'estimated_seconds': estimated_seconds,
            'estimated_minutes': estimated_seconds / 60,
            'cells_per_second': cell_count / max(estimated_seconds, 0.1),
            'complexity_factor': multiplier
        }
        
    def get_memory_requirements(self, 
                              data_size_mb: float,
                              cell_count: int) -> Dict[str, float]:
        """
        Estimate memory requirements for feature extraction.
        
        Args:
            data_size_mb: Size of input data in MB
            cell_count: Number of cells to process
            
        Returns:
            Dictionary with memory estimates in MB
        """
        # Base memory estimates
        input_memory = data_size_mb
        
        # Working memory (varies by feature type)
        working_multiplier = {
            'texture': 3.0,      # Texture analysis needs significant working memory
            'spectral': 2.0,     # Spectral indices need intermediate calculations
            'species_richness': 1.5,  # Species data processing
            'climate': 1.2,      # Climate interpolation
            'elevation': 1.0     # Simple elevation processing
        }.get(self.feature_type, 1.5)
        
        working_memory = input_memory * working_multiplier
        
        # Output memory (results storage)
        output_memory = cell_count * 0.001  # ~1KB per cell for results
        
        # Total with overhead
        total_memory = (input_memory + working_memory + output_memory) * 1.2  # 20% overhead
        
        return {
            'input_memory_mb': input_memory,
            'working_memory_mb': working_memory,
            'output_memory_mb': output_memory,
            'total_memory_mb': total_memory,
            'recommended_batch_size': max(1, int(1000 / (total_memory / cell_count))) if cell_count > 0 else 1000
        }
        
    def create_feature_metadata(self, 
                              computation_time: float,
                              memory_used: float,
                              **kwargs) -> Dict[str, Any]:
        """
        Create comprehensive metadata for feature results.
        
        Args:
            computation_time: Time taken for computation in seconds
            memory_used: Memory used in MB
            **kwargs: Additional metadata
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'feature_extractor': self.__class__.__name__,
            'feature_type': self.feature_type,
            'source_type': self.source_type.value,
            'computation_time_seconds': computation_time,
            'memory_used_mb': memory_used,
            'normalized': self.normalize,
            'config': self.config,
            **kwargs
        }
        
        # Add multi-band info if applicable
        if self.supports_multi_band():
            metadata['supports_multi_band'] = True
            metadata['required_bands'] = self.required_raster_bands
            
        return metadata