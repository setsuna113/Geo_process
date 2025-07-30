"""Machine Learning analyzer interface extending base analyzer capabilities."""

from abc import abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from .analyzer import IAnalyzer, AnalysisResult


class IMLAnalyzer(IAnalyzer):
    """
    Interface for machine learning analyzers.
    
    Extends the base analyzer interface with ML-specific methods for
    training, prediction, and model persistence.
    """
    
    @abstractmethod
    def fit(self, 
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            sample_weight: Optional[np.ndarray] = None,
            **kwargs) -> 'IMLAnalyzer':
        """
        Train the model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target values
            sample_weight: Optional sample weights
            **kwargs: Additional model-specific parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, 
                X: Union[np.ndarray, pd.DataFrame],
                **kwargs) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def fit_predict(self,
                    X: Union[np.ndarray, pd.DataFrame],
                    y: Union[np.ndarray, pd.Series],
                    sample_weight: Optional[np.ndarray] = None,
                    **kwargs) -> np.ndarray:
        """
        Fit the model and return predictions on the training data.
        
        Args:
            X: Feature matrix
            y: Target values
            sample_weight: Optional sample weights
            **kwargs: Additional parameters
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self,
                 X: Union[np.ndarray, pd.DataFrame],
                 y: Union[np.ndarray, pd.Series],
                 metrics: Optional[List[str]] = None,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Feature matrix
            y: True target values
            metrics: List of metrics to compute
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of metric names to values
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if available.
        
        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not available
        """
        pass
    
    @abstractmethod
    def save_model(self, path: Path) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, path: Path) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load the model from
        """
        pass
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dictionary of parameter names to values
        """
        pass
    
    @abstractmethod
    def set_model_params(self, **params) -> 'IMLAnalyzer':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def cross_validate(self,
                      X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, pd.Series],
                      cv_strategy: Any,
                      metrics: Optional[List[str]] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Perform cross-validation with the specified strategy.
        
        Args:
            X: Feature matrix
            y: Target values
            cv_strategy: Cross-validation strategy object
            metrics: Metrics to compute
            **kwargs: Additional CV parameters
            
        Returns:
            Dictionary with CV results including per-fold metrics
        """
        pass
    
    @abstractmethod
    def explain_predictions(self,
                           X: Union[np.ndarray, pd.DataFrame],
                           method: str = "shap",
                           **kwargs) -> Dict[str, Any]:
        """
        Explain model predictions using specified method.
        
        Args:
            X: Feature matrix to explain
            method: Explanation method ("shap", "lime", etc.)
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary with explanation results
        """
        pass