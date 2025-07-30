"""Feature builder interface for machine learning feature engineering."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Set, Optional, Union, Tuple
import numpy as np
import pandas as pd


class IFeatureBuilder(ABC):
    """
    Interface for feature builders that create ML features from raw data.
    
    Feature builders are responsible for transforming raw biodiversity data
    into features suitable for machine learning models.
    """
    
    @abstractmethod
    def build_features(self,
                      data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                      **kwargs) -> pd.DataFrame:
        """
        Build features from input data.
        
        Args:
            data: Input data as DataFrame or dictionary of arrays
            **kwargs: Additional parameters for feature building
            
        Returns:
            DataFrame with engineered features
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names that will be created.
        
        Returns:
            List of feature column names
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> Set[str]:
        """
        Get set of required input columns.
        
        Returns:
            Set of column names required in input data
        """
        pass
    
    @abstractmethod
    def get_feature_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata about each feature.
        
        Returns:
            Dictionary mapping feature names to their metadata:
            - description: Human-readable description
            - dtype: Expected data type
            - range: Expected value range (min, max)
            - category: Feature category (spatial, ecological, etc.)
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: Union[pd.DataFrame, Dict[str, np.ndarray]]) -> Tuple[bool, List[str]]:
        """
        Validate that input data has required columns and correct format.
        
        Args:
            data: Input data to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        pass
    
    @abstractmethod
    def fit(self, data: Union[pd.DataFrame, Dict[str, np.ndarray]], **kwargs) -> 'IFeatureBuilder':
        """
        Fit the feature builder on training data if needed.
        
        Some feature builders may need to learn parameters from training data
        (e.g., scaling parameters, binning boundaries).
        
        Args:
            data: Training data
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, data: Union[pd.DataFrame, Dict[str, np.ndarray]], **kwargs) -> pd.DataFrame:
        """
        Transform data using fitted parameters.
        
        Args:
            data: Data to transform
            **kwargs: Additional transformation parameters
            
        Returns:
            Transformed features as DataFrame
        """
        pass
    
    @abstractmethod
    def fit_transform(self, data: Union[pd.DataFrame, Dict[str, np.ndarray]], **kwargs) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            data: Data to fit and transform
            **kwargs: Additional parameters
            
        Returns:
            Transformed features as DataFrame
        """
        pass
    
    @abstractmethod
    def get_feature_importance_prior(self) -> Optional[Dict[str, float]]:
        """
        Get prior importance/weight for features if known.
        
        Some features may have known importance based on domain knowledge.
        
        Returns:
            Dictionary mapping feature names to importance scores,
            or None if no prior importance
        """
        pass


class ICompositeFeatureBuilder(IFeatureBuilder):
    """
    Interface for composite feature builders that combine multiple builders.
    """
    
    @abstractmethod
    def add_builder(self, builder: IFeatureBuilder, prefix: Optional[str] = None) -> None:
        """
        Add a feature builder to the composite.
        
        Args:
            builder: Feature builder to add
            prefix: Optional prefix for feature names to avoid conflicts
        """
        pass
    
    @abstractmethod
    def remove_builder(self, builder_name: str) -> None:
        """
        Remove a feature builder by name.
        
        Args:
            builder_name: Name of builder to remove
        """
        pass
    
    @abstractmethod
    def get_builders(self) -> Dict[str, IFeatureBuilder]:
        """
        Get all registered builders.
        
        Returns:
            Dictionary mapping builder names to instances
        """
        pass
    
    @abstractmethod
    def get_feature_provenance(self) -> Dict[str, str]:
        """
        Get mapping of feature names to their source builders.
        
        Returns:
            Dictionary mapping feature names to builder names
        """
        pass