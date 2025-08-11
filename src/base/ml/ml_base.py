"""Unified base class for all machine learning components.

This provides a single, coherent base class hierarchy for the ML pipeline.
"""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of machine learning models."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    GLM = "glm"
    GBM = "gbm"
    DML = "double_ml"


class BaseMLComponent(ABC):
    """Unified base class for all ML components in the biodiversity framework.
    
    This class provides:
    - Unified configuration handling
    - Model diagnostics interface
    - Result summarization
    - Consistent interface for all ML components
    - Type-safe methods
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_type: ModelType = ModelType.REGRESSION,
        name: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize base ML component.
        
        Args:
            config: Configuration dictionary
            model_type: Type of model
            name: Optional component name
            **kwargs: Additional parameters stored as attributes
        """
        self.config: Dict[str, Any] = config or {}
        self.model_type: ModelType = model_type
        self.name: str = name or self.__class__.__name__
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Model and data storage
        self.model: Optional[Any] = None
        self.data: Optional[Any] = None
        self.results: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}
        
        logger.info(f"Initialized {self.name} with model type {model_type.value}")
    
    @abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None, **kwargs) -> 'BaseMLComponent':
        """Fit the model.
        
        Args:
            X: Features
            y: Target (optional for unsupervised methods)
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: Any, **kwargs) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions
        """
        pass
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this component.
        
        Args:
            deep: If True, return parameters for sub-objects
            
        Returns:
            Parameter dictionary
        """
        params = {
            'config': self.config,
            'model_type': self.model_type,
            'name': self.name
        }
        return params
    
    def set_params(self, **params) -> 'BaseMLComponent':
        """Set parameters for this component.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in self.config:
                self.config[key] = value
        return self
    
    def validate_input(
        self, 
        X: Any, 
        y: Optional[Any] = None,
        require_fitted: bool = False
    ) -> Tuple[bool, List[str]]:
        """Validate input data.
        
        Args:
            X: Features to validate
            y: Target to validate (optional)
            require_fitted: Whether model must be fitted
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors: List[str] = []
        
        if require_fitted and self.model is None:
            errors.append("Model not fitted yet")
        
        if X is None:
            errors.append("Features (X) cannot be None")
        
        # Add more validation as needed
        
        return len(errors) == 0, errors
    
    def get_metrics(self) -> Dict[str, float]:
        """Get model metrics.
        
        Returns:
            Dictionary of metric names to values
        """
        return self.metrics.copy()
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available.
        
        Returns:
            Dictionary of feature names to importance scores
        """
        # Default implementation - override in subclasses
        return None
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        # Default implementation - override in subclasses
        logger.warning(f"{self.name}: Save not implemented")
    
    def load(self, path: Union[str, Path]) -> None:
        """Load model from disk.
        
        Args:
            path: Path to load model from
        """
        # Default implementation - override in subclasses
        logger.warning(f"{self.name}: Load not implemented")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}(model_type={self.model_type.value}, fitted={self.model is not None})"