"""Linear regression model implementation with Ridge regularization."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging

from ...base.ml_analyzer import BaseMLAnalyzer
from ...abstractions.types.ml_types import ModelType
from ...core.registry import ml_model

logger = logging.getLogger(__name__)


@ml_model(
    model_type="regression",
    requires_scaling=True,
    handles_missing_values=False,
    description="Ridge regression model for biodiversity richness prediction"
)
class LinearRegressionAnalyzer(BaseMLAnalyzer):
    """
    Linear regression analyzer using Ridge regularization.
    
    Always uses Ridge (L2 regularization) to handle potential collinearity
    in biodiversity features. Requires feature scaling for optimal performance.
    """
    
    def __init__(self, 
                 config: Union[Dict[str, Any], Any],
                 db_connection: Optional[Any] = None,
                 **kwargs):
        """Initialize linear regression analyzer."""
        super().__init__(config, db_connection, model_type=ModelType.REGRESSION, **kwargs)
        
        # Get model configuration
        model_config = self.safe_get_config('machine_learning.models.linear_regression', {})
        
        # Initialize model with config parameters
        self.alpha = model_config.get('alpha', 1.0)
        self.fit_intercept = model_config.get('fit_intercept', True)
        self.max_iter = model_config.get('max_iter', 1000)
        self.solver = model_config.get('solver', 'auto')
        
        # Initialize Ridge model
        self.model = Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=42
        )
        
        # Scaler for features
        scaling_config = self.safe_get_config('machine_learning.preprocessing.scaling', {})
        self.scaling_method = scaling_config.get('method', 'standard')
        self.scaler = StandardScaler() if self.scaling_method == 'standard' else None
        
        # Feature importance (coefficients)
        self.coefficients_ = None
        
    def fit(self, 
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            sample_weight: Optional[np.ndarray] = None,
            **kwargs) -> 'LinearRegressionAnalyzer':
        """
        Fit the Ridge regression model.
        
        Args:
            X: Feature matrix
            y: Target values
            sample_weight: Optional sample weights
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        # Convert to numpy if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Scale features
        if self.scaler is not None:
            logger.info("Scaling features using StandardScaler")
            X_scaled = self.scaler.fit_transform(X_array)
        else:
            X_scaled = X_array
        
        # Fit model
        logger.info(f"Fitting Ridge regression with alpha={self.alpha}")
        self.model.fit(X_scaled, y_array, sample_weight=sample_weight)
        
        # Store coefficients
        self.coefficients_ = self.model.coef_
        self.is_fitted = True
        
        # Log model info
        logger.info(f"Model fitted with {len(self.coefficients_)} features")
        
        return self
    
    def predict(self, 
                X: Union[np.ndarray, pd.DataFrame],
                **kwargs) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Feature matrix
            **kwargs: Additional parameters
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_array)
        else:
            X_scaled = X_array
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Ensure non-negative predictions for richness
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def evaluate(self,
                 X: Union[np.ndarray, pd.DataFrame],
                 y: Union[np.ndarray, pd.Series],
                 metrics: Optional[List[str]] = None,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True target values
            metrics: List of metrics to compute
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of metric names to values
        """
        if metrics is None:
            metrics = ['r2', 'rmse', 'mae', 'mape']
        
        # Get predictions
        y_pred = self.predict(X)
        y_true = y.values if isinstance(y, pd.Series) else y
        
        results = {}
        
        # Compute requested metrics
        if 'r2' in metrics:
            results['r2'] = r2_score(y_true, y_pred)
        
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(y_true, y_pred)
        
        if 'mape' in metrics:
            # Mean Absolute Percentage Error (avoid division by zero)
            mask = y_true != 0
            if np.any(mask):
                results['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                results['mape'] = np.nan
        
        # Add residual statistics
        residuals = y_true - y_pred
        results['residual_mean'] = np.mean(residuals)
        results['residual_std'] = np.std(residuals)
        
        return results
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance as absolute coefficient values.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted or self.coefficients_ is None:
            return None
        
        if self.feature_names is None:
            return None
        
        # Use absolute values of coefficients as importance
        importances = np.abs(self.coefficients_)
        
        # Normalize to sum to 1
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        # Create dictionary
        importance_dict = {
            name: float(importance) 
            for name, importance in zip(self.feature_names, importances)
        }
        
        # Sort by importance
        importance_dict = dict(sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return importance_dict
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters."""
        return {
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'max_iter': self.max_iter,
            'solver': self.solver,
            'scaling_method': self.scaling_method,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'is_fitted': self.is_fitted
        }
    
    def set_model_params(self, **params) -> 'LinearRegressionAnalyzer':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        # Update Ridge parameters
        ridge_params = {}
        if 'alpha' in params:
            self.alpha = params['alpha']
            ridge_params['alpha'] = self.alpha
        
        if 'fit_intercept' in params:
            self.fit_intercept = params['fit_intercept']
            ridge_params['fit_intercept'] = self.fit_intercept
        
        if 'max_iter' in params:
            self.max_iter = params['max_iter']
            ridge_params['max_iter'] = self.max_iter
        
        if 'solver' in params:
            self.solver = params['solver']
            ridge_params['solver'] = self.solver
        
        # Update model if parameters changed
        if ridge_params:
            self.model.set_params(**ridge_params)
        
        # Update scaling method
        if 'scaling_method' in params:
            self.scaling_method = params['scaling_method']
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = None
        
        return self
    
    def get_coefficients(self) -> Optional[Dict[str, float]]:
        """
        Get model coefficients (not absolute values).
        
        Returns:
            Dictionary mapping feature names to coefficient values
        """
        if not self.is_fitted or self.coefficients_ is None:
            return None
        
        if self.feature_names is None:
            return None
        
        # Create dictionary with actual coefficient values
        coef_dict = {
            name: float(coef) 
            for name, coef in zip(self.feature_names, self.coefficients_)
        }
        
        # Add intercept
        if self.fit_intercept:
            coef_dict['_intercept'] = float(self.model.intercept_)
        
        return coef_dict
    
    def explain_predictions(self,
                           X: Union[np.ndarray, pd.DataFrame],
                           method: str = "coefficients",
                           **kwargs) -> Dict[str, Any]:
        """
        Explain predictions using linear coefficients.
        
        Args:
            X: Feature matrix to explain
            method: Explanation method
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with explanation results
        """
        if method == "coefficients":
            return {
                'method': 'linear_coefficients',
                'coefficients': self.get_coefficients(),
                'feature_importance': self.get_feature_importance(),
                'model_type': 'linear'
            }
        else:
            # Fall back to base implementation
            return super().explain_predictions(X, method, **kwargs)