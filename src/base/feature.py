"""Base feature extractor class."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import logging

from ..core.registry import component_registry
from ..config import config
from ..database.schema import schema

logger = logging.getLogger(__name__)

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