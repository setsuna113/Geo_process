"""Base implementation for feature builders."""

from abc import abstractmethod
from typing import Dict, Any, List, Set, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field

from ..abstractions.interfaces.feature_builder import IFeatureBuilder, ICompositeFeatureBuilder
from ..abstractions.types.ml_types import FeatureMetadata

logger = logging.getLogger(__name__)


class BaseFeatureBuilder(IFeatureBuilder):
    """
    Base implementation for feature builders.
    
    Provides common functionality for feature engineering including:
    - Input validation
    - Progress tracking integration
    - Feature metadata management
    - Transformation pipeline support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature builder.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.is_fitted = False
        self.feature_metadata = {}
        self._fitted_params = {}
        
        # Progress callback
        self._progress_callback = None
        
    def build_features(self,
                      data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                      **kwargs) -> pd.DataFrame:
        """
        Build features from input data.
        
        Default implementation that calls fit_transform.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with engineered features
        """
        return self.fit_transform(data, **kwargs)
    
    def fit(self, data: Union[pd.DataFrame, Dict[str, np.ndarray]], **kwargs) -> 'BaseFeatureBuilder':
        """
        Fit the feature builder on training data.
        
        Default implementation for builders that don't need fitting.
        
        Args:
            data: Training data
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        # Validate input
        is_valid, issues = self.validate_input(data)
        if not is_valid:
            raise ValueError(f"Input validation failed: {', '.join(issues)}")
        
        # Let subclasses implement actual fitting
        self._fit(data, **kwargs)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[pd.DataFrame, Dict[str, np.ndarray]], **kwargs) -> pd.DataFrame:
        """
        Transform data using fitted parameters.
        
        Args:
            data: Data to transform
            **kwargs: Additional parameters
            
        Returns:
            Transformed features as DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Feature builder must be fitted before transform")
        
        # Validate input
        is_valid, issues = self.validate_input(data)
        if not is_valid:
            raise ValueError(f"Input validation failed: {', '.join(issues)}")
        
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # Let subclasses implement actual transformation
        features = self._transform(data, **kwargs)
        
        # Ensure result is DataFrame
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        
        # Add feature metadata
        self._update_feature_metadata(features)
        
        return features
    
    def fit_transform(self, data: Union[pd.DataFrame, Dict[str, np.ndarray]], **kwargs) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            data: Data to fit and transform
            **kwargs: Additional parameters
            
        Returns:
            Transformed features as DataFrame
        """
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)
    
    def validate_input(self, data: Union[pd.DataFrame, Dict[str, np.ndarray]]) -> Tuple[bool, List[str]]:
        """
        Validate that input data has required columns and correct format.
        
        Args:
            data: Input data to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Convert to DataFrame for validation
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Check required columns
        required = self.get_required_columns()
        missing = required - set(df.columns)
        if missing:
            issues.append(f"Missing required columns: {missing}")
        
        # Check for empty data
        if len(df) == 0:
            issues.append("Input data is empty")
        
        # Let subclasses add specific validation
        subclass_issues = self._validate_input_specific(df)
        issues.extend(subclass_issues)
        
        return len(issues) == 0, issues
    
    def get_feature_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata about each feature.
        
        Returns:
            Dictionary mapping feature names to their metadata
        """
        return self.feature_metadata
    
    def get_feature_importance_prior(self) -> Optional[Dict[str, float]]:
        """
        Get prior importance/weight for features if known.
        
        Default implementation returns None.
        Subclasses can override to provide domain knowledge.
        
        Returns:
            Dictionary mapping feature names to importance scores,
            or None if no prior importance
        """
        return None
    
    def set_progress_callback(self, callback: callable) -> None:
        """
        Set callback for progress updates.
        
        Args:
            callback: Function that accepts (current, total, message)
        """
        self._progress_callback = callback
    
    def _update_progress(self, current: int, total: int, message: str) -> None:
        """Update progress through callback if set."""
        if self._progress_callback:
            try:
                self._progress_callback(current, total, message)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def _update_feature_metadata(self, features: pd.DataFrame) -> None:
        """Update feature metadata based on computed features."""
        for col in features.columns:
            if col not in self.feature_metadata:
                self.feature_metadata[col] = {
                    'name': col,
                    'description': f"Feature {col}",
                    'category': 'unknown',
                    'dtype': str(features[col].dtype),
                    'min_value': float(features[col].min()) if pd.api.types.is_numeric_dtype(features[col]) else None,
                    'max_value': float(features[col].max()) if pd.api.types.is_numeric_dtype(features[col]) else None,
                    'mean_value': float(features[col].mean()) if pd.api.types.is_numeric_dtype(features[col]) else None,
                    'std_value': float(features[col].std()) if pd.api.types.is_numeric_dtype(features[col]) else None,
                    'missing_rate': float(features[col].isna().mean())
                }
    
    # Abstract methods that subclasses must implement
    
    @abstractmethod
    def _fit(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Internal fit method to be implemented by subclasses.
        
        Args:
            data: Training data as DataFrame
            **kwargs: Additional parameters
        """
        pass
    
    @abstractmethod
    def _transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Internal transform method to be implemented by subclasses.
        
        Args:
            data: Data to transform as DataFrame
            **kwargs: Additional parameters
            
        Returns:
            Transformed features as DataFrame
        """
        pass
    
    def _validate_input_specific(self, data: pd.DataFrame) -> List[str]:
        """
        Subclass-specific validation.
        
        Override in subclasses to add specific validation logic.
        
        Args:
            data: Input data as DataFrame
            
        Returns:
            List of validation issues
        """
        return []
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names that will be created."""
        pass
    
    @abstractmethod
    def get_required_columns(self) -> Set[str]:
        """Get set of required input columns."""
        pass


class BaseCompositeFeatureBuilder(BaseFeatureBuilder, ICompositeFeatureBuilder):
    """
    Base implementation for composite feature builders.
    
    Combines multiple feature builders into a single pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize composite feature builder."""
        super().__init__(config)
        self.builders = {}
        self.builder_prefixes = {}
        self.feature_provenance = {}
    
    def add_builder(self, builder: IFeatureBuilder, prefix: Optional[str] = None) -> None:
        """
        Add a feature builder to the composite.
        
        Args:
            builder: Feature builder to add
            prefix: Optional prefix for feature names
        """
        builder_name = builder.__class__.__name__
        
        # Ensure unique names
        if builder_name in self.builders:
            i = 1
            while f"{builder_name}_{i}" in self.builders:
                i += 1
            builder_name = f"{builder_name}_{i}"
        
        self.builders[builder_name] = builder
        self.builder_prefixes[builder_name] = prefix
        
        # Update feature provenance
        for feature in builder.get_feature_names():
            prefixed_name = f"{prefix}_{feature}" if prefix else feature
            self.feature_provenance[prefixed_name] = builder_name
    
    def remove_builder(self, builder_name: str) -> None:
        """
        Remove a feature builder by name.
        
        Args:
            builder_name: Name of builder to remove
        """
        if builder_name in self.builders:
            # Remove from provenance
            features_to_remove = [
                f for f, b in self.feature_provenance.items() 
                if b == builder_name
            ]
            for feature in features_to_remove:
                del self.feature_provenance[feature]
            
            # Remove builder
            del self.builders[builder_name]
            del self.builder_prefixes[builder_name]
    
    def get_builders(self) -> Dict[str, IFeatureBuilder]:
        """Get all registered builders."""
        return self.builders.copy()
    
    def get_feature_provenance(self) -> Dict[str, str]:
        """Get mapping of feature names to their source builders."""
        return self.feature_provenance.copy()
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names from all builders."""
        feature_names = []
        for builder_name, builder in self.builders.items():
            prefix = self.builder_prefixes[builder_name]
            for feature in builder.get_feature_names():
                prefixed_name = f"{prefix}_{feature}" if prefix else feature
                feature_names.append(prefixed_name)
        return feature_names
    
    def get_required_columns(self) -> Set[str]:
        """Get union of required columns from all builders."""
        required = set()
        for builder in self.builders.values():
            required.update(builder.get_required_columns())
        return required
    
    def _fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit all builders."""
        total_builders = len(self.builders)
        for i, (name, builder) in enumerate(self.builders.items()):
            self._update_progress(i, total_builders, f"Fitting {name}")
            builder.fit(data, **kwargs)
    
    def _transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform with all builders and combine results."""
        all_features = []
        total_builders = len(self.builders)
        
        for i, (builder_name, builder) in enumerate(self.builders.items()):
            self._update_progress(i, total_builders, f"Transforming with {builder_name}")
            
            # Transform with this builder
            features = builder.transform(data, **kwargs)
            
            # Apply prefix if specified
            prefix = self.builder_prefixes[builder_name]
            if prefix:
                features = features.add_prefix(f"{prefix}_")
            
            all_features.append(features)
        
        # Combine all features
        if all_features:
            combined = pd.concat(all_features, axis=1)
            
            # Handle duplicate columns
            duplicate_cols = combined.columns[combined.columns.duplicated()]
            if len(duplicate_cols) > 0:
                logger.warning(f"Duplicate feature names found: {duplicate_cols}")
                # Make unique by adding suffix
                combined = combined.loc[:, ~combined.columns.duplicated()]
            
            return combined
        else:
            return pd.DataFrame(index=data.index)
    
    def get_feature_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get combined metadata from all builders."""
        all_metadata = {}
        
        for builder_name, builder in self.builders.items():
            prefix = self.builder_prefixes[builder_name]
            builder_metadata = builder.get_feature_metadata()
            
            for feature, metadata in builder_metadata.items():
                prefixed_name = f"{prefix}_{feature}" if prefix else feature
                all_metadata[prefixed_name] = metadata.copy()
                all_metadata[prefixed_name]['source_builder'] = builder_name
        
        return all_metadata