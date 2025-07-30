"""LightGBM regression model implementation."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
import logging
import warnings

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not installed. LightGBMAnalyzer will not be available.")

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ...base.ml_analyzer import BaseMLAnalyzer
from ...abstractions.types.ml_types import ModelType
from ...core.registry import ml_model

logger = logging.getLogger(__name__)


if LIGHTGBM_AVAILABLE:
    @ml_model(
        model_type="regression",
        requires_scaling=False,  # Tree-based models don't need scaling
        handles_missing_values=True,  # LightGBM handles missing values natively
        description="LightGBM gradient boosting for biodiversity richness prediction"
    )
    class LightGBMAnalyzer(BaseMLAnalyzer):
        """
        LightGBM analyzer for gradient boosting regression.
        
        Handles missing values natively and doesn't require feature scaling.
        Provides built-in regularization and feature importance.
        """
        
        def __init__(self, 
                     config: Union[Dict[str, Any], Any],
                     db_connection: Optional[Any] = None,
                     **kwargs):
            """Initialize LightGBM analyzer."""
            super().__init__(config, db_connection, model_type=ModelType.REGRESSION, **kwargs)
            
            # Get model configuration
            model_config = self.safe_get_config('machine_learning.models.lightgbm', {})
            
            # Core parameters
            self.num_leaves = model_config.get('num_leaves', 31)
            self.learning_rate = model_config.get('learning_rate', 0.1)
            self.n_estimators = model_config.get('n_estimators', 100)
            
            # Regularization
            self.reg_alpha = model_config.get('reg_alpha', 0.0)  # L1
            self.reg_lambda = model_config.get('reg_lambda', 0.0)  # L2
            
            # Tree constraints
            self.min_child_samples = model_config.get('min_child_samples', 20)
            self.subsample = model_config.get('subsample', 0.8)
            self.colsample_bytree = model_config.get('colsample_bytree', 0.8)
            
            # Other parameters
            self.objective = model_config.get('objective', 'regression')
            self.metric = model_config.get('metric', 'rmse')
            self.random_state = model_config.get('random_state', 42)
            self.n_jobs = model_config.get('n_jobs', -1)
            self.verbose = model_config.get('verbose', -1)
            
            # Build parameter dictionary
            self.lgb_params = {
                'objective': self.objective,
                'metric': self.metric,
                'num_leaves': self.num_leaves,
                'learning_rate': self.learning_rate,
                'n_estimators': self.n_estimators,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'min_child_samples': self.min_child_samples,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'verbose': self.verbose,
                'force_col_wise': True  # For better performance
            }
            
            # Initialize model
            self.model = lgb.LGBMRegressor(**self.lgb_params)
            
            # Training history
            self.evals_result_ = {}
            self.best_iteration_ = None
            
        def fit(self, 
                X: Union[np.ndarray, pd.DataFrame],
                y: Union[np.ndarray, pd.Series],
                sample_weight: Optional[np.ndarray] = None,
                eval_set: Optional[List[tuple]] = None,
                early_stopping_rounds: Optional[int] = 10,
                **kwargs) -> 'LightGBMAnalyzer':
            """
            Fit the LightGBM model.
            
            Args:
                X: Feature matrix
                y: Target values
                sample_weight: Optional sample weights
                eval_set: Validation set for early stopping
                early_stopping_rounds: Rounds for early stopping
                **kwargs: Additional parameters
                
            Returns:
                Self for method chaining
            """
            # Store feature names if DataFrame
            if isinstance(X, pd.DataFrame):
                self.feature_names = list(X.columns)
            
            # Prepare callbacks
            callbacks = []
            if self._progress_callback:
                # Create LightGBM callback for progress
                def lgb_progress(env):
                    if env.iteration % 10 == 0:  # Update every 10 iterations
                        self._update_progress(
                            env.iteration, 
                            env.end_iteration, 
                            f"Training iteration {env.iteration}/{env.end_iteration}"
                        )
                callbacks.append(lgb_progress)
            
            # Log training start
            logger.info(f"Training LightGBM with {self.n_estimators} trees")
            
            # Fit model
            fit_params = {
                'sample_weight': sample_weight,
                'callbacks': callbacks
            }
            
            # Add early stopping if validation set provided
            if eval_set is not None:
                fit_params['eval_set'] = eval_set
                fit_params['eval_names'] = ['train', 'valid'] if len(eval_set) > 1 else ['valid']
                fit_params['eval_metric'] = self.metric
                if early_stopping_rounds is not None:
                    logger.info(f"Using early stopping with patience={early_stopping_rounds}")
            
            self.model.fit(X, y, **fit_params)
            
            # Store training results
            if hasattr(self.model, 'evals_result_'):
                self.evals_result_ = self.model.evals_result_
            if hasattr(self.model, 'best_iteration_'):
                self.best_iteration_ = self.model.best_iteration_
                logger.info(f"Best iteration: {self.best_iteration_}")
            
            self.is_fitted = True
            
            # Log feature importance summary
            importance = self.get_feature_importance()
            if importance:
                top_features = list(importance.keys())[:5]
                logger.info(f"Top 5 features: {top_features}")
            
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
            
            # Make predictions
            predictions = self.model.predict(X, **kwargs)
            
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
                # Mean Absolute Percentage Error
                mask = y_true != 0
                if np.any(mask):
                    results['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                else:
                    results['mape'] = np.nan
            
            # Add residual statistics
            residuals = y_true - y_pred
            results['residual_mean'] = np.mean(residuals)
            results['residual_std'] = np.std(residuals)
            
            # Add training metrics if available
            if self.best_iteration_ is not None:
                results['best_iteration'] = self.best_iteration_
                results['actual_iterations'] = self.model.n_estimators
            
            return results
        
        def get_feature_importance(self) -> Optional[Dict[str, float]]:
            """
            Get feature importance scores from the model.
            
            Returns:
                Dictionary mapping feature names to importance scores
            """
            if not self.is_fitted:
                return None
            
            # Get importance values
            importance_values = self.model.feature_importances_
            
            # Get feature names
            if self.feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importance_values))]
            else:
                feature_names = self.feature_names
            
            # Normalize importances to sum to 1
            if np.sum(importance_values) > 0:
                importance_values = importance_values / np.sum(importance_values)
            
            # Create dictionary
            importance_dict = {
                name: float(importance) 
                for name, importance in zip(feature_names, importance_values)
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
            params = {
                'num_leaves': self.num_leaves,
                'learning_rate': self.learning_rate,
                'n_estimators': self.n_estimators,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'min_child_samples': self.min_child_samples,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'objective': self.objective,
                'metric': self.metric,
                'is_fitted': self.is_fitted
            }
            
            if self.best_iteration_ is not None:
                params['best_iteration'] = self.best_iteration_
            
            return params
        
        def set_model_params(self, **params) -> 'LightGBMAnalyzer':
            """
            Set model parameters.
            
            Args:
                **params: Parameters to set
                
            Returns:
                Self for method chaining
            """
            # Update internal parameters
            lgb_params = {}
            
            param_mapping = {
                'num_leaves': 'num_leaves',
                'learning_rate': 'learning_rate',
                'n_estimators': 'n_estimators',
                'reg_alpha': 'reg_alpha',
                'reg_lambda': 'reg_lambda',
                'min_child_samples': 'min_child_samples',
                'subsample': 'subsample',
                'colsample_bytree': 'colsample_bytree',
                'objective': 'objective',
                'metric': 'metric'
            }
            
            for param_name, lgb_name in param_mapping.items():
                if param_name in params:
                    setattr(self, param_name, params[param_name])
                    lgb_params[lgb_name] = params[param_name]
            
            # Update model if parameters changed
            if lgb_params:
                self.lgb_params.update(lgb_params)
                self.model.set_params(**lgb_params)
            
            return self
        
        def get_tree_splits(self) -> Optional[Dict[str, int]]:
            """
            Get number of times each feature was used for splitting.
            
            Returns:
                Dictionary mapping feature names to split counts
            """
            if not self.is_fitted:
                return None
            
            # Get split counts (gain-based importance)
            split_counts = self.model.booster_.feature_importance(importance_type='split')
            
            # Get feature names
            if self.feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(split_counts))]
            else:
                feature_names = self.feature_names
            
            # Create dictionary
            splits_dict = {
                name: int(count) 
                for name, count in zip(feature_names, split_counts)
                if count > 0  # Only include features that were actually used
            }
            
            # Sort by count
            splits_dict = dict(sorted(
                splits_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return splits_dict
        
        def explain_predictions(self,
                               X: Union[np.ndarray, pd.DataFrame],
                               method: str = "feature_importance",
                               **kwargs) -> Dict[str, Any]:
            """
            Explain predictions using various methods.
            
            Args:
                X: Feature matrix to explain
                method: Explanation method
                **kwargs: Additional parameters
                
            Returns:
                Dictionary with explanation results
            """
            if method == "feature_importance":
                return {
                    'method': 'feature_importance',
                    'importance_gain': self.get_feature_importance(),
                    'importance_split': self.get_tree_splits(),
                    'model_type': 'tree_ensemble',
                    'n_trees': self.model.n_estimators if self.is_fitted else 0
                }
            elif method == "shap" and kwargs.get('use_shap', False):
                # SHAP integration would go here if shap is installed
                # For now, fall back to feature importance
                logger.warning("SHAP not implemented, using feature importance instead")
                return self.explain_predictions(X, method="feature_importance")
            else:
                # Fall back to base implementation
                return super().explain_predictions(X, method, **kwargs)