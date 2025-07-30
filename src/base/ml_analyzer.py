"""Base implementation for machine learning analyzers."""

from abc import abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import pickle
import json
from datetime import datetime

from .analyzer import BaseAnalyzer
from ..abstractions.interfaces.ml_analyzer import IMLAnalyzer
from ..abstractions.types.ml_types import (
    MLResult, ModelMetadata, CVResult, CVFold,
    ModelType, ValidationResult
)
from ..abstractions.interfaces.analyzer import AnalysisResult, AnalysisMetadata

logger = logging.getLogger(__name__)


class BaseMLAnalyzer(BaseAnalyzer, IMLAnalyzer):
    """
    Base implementation for ML analyzers extending the base analyzer.
    
    Provides common functionality for all ML models including:
    - Data preparation and validation
    - Model persistence
    - Feature importance tracking
    - Cross-validation support
    - Integration with progress tracking and checkpointing
    """
    
    def __init__(self, 
                 config: Union[Dict[str, Any], Any],
                 db_connection: Optional[Any] = None,
                 model_type: ModelType = ModelType.REGRESSION,
                 **kwargs):
        """
        Initialize ML analyzer.
        
        Args:
            config: Configuration dict or object
            db_connection: Optional database connection
            model_type: Type of ML model
            **kwargs: Additional parameters
        """
        super().__init__(config, db_connection, **kwargs)
        
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.target_name = None
        self.model_metadata = None
        
        # ML-specific configuration
        self.save_predictions = self.safe_get_config('machine_learning.output.save_predictions', True)
        self.save_model = self.safe_get_config('machine_learning.output.save_model', True)
        self.output_formats = self.safe_get_config('machine_learning.output.formats', ['parquet', 'pickle'])
        
    def analyze(self, 
                data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                target_column: str = 'target',
                feature_columns: Optional[List[str]] = None,
                cv_strategy: Optional[Any] = None,
                **kwargs) -> AnalysisResult:
        """
        Perform ML analysis by fitting and evaluating the model.
        
        Args:
            data: Input data with features and target
            target_column: Name of target column
            feature_columns: List of feature columns (None = all except target)
            cv_strategy: Optional CV strategy for evaluation
            **kwargs: Additional parameters
            
        Returns:
            MLResult with predictions and metrics
        """
        start_time = time.time()
        
        # Prepare data
        self._update_progress(0, 100, "Preparing data for ML analysis")
        X, y, feature_names = self._prepare_ml_data(data, target_column, feature_columns)
        
        # Store feature and target names
        self.feature_names = feature_names
        self.target_name = target_column
        
        # Validate data
        self._update_progress(10, 100, "Validating input data")
        is_valid, issues = self.validate_ml_data(X, y)
        if not is_valid:
            raise ValueError(f"Data validation failed: {', '.join(issues)}")
        
        # Perform cross-validation if strategy provided
        cv_results = None
        if cv_strategy is not None:
            self._update_progress(20, 100, "Performing cross-validation")
            cv_results = self.cross_validate(X, y, cv_strategy, **kwargs)
        
        # Fit model on full data
        self._update_progress(60, 100, "Training model on full dataset")
        self.fit(X, y, **kwargs)
        
        # Make predictions
        self._update_progress(80, 100, "Generating predictions")
        predictions = self.predict(X)
        
        # Evaluate on training data
        self._update_progress(90, 100, "Computing metrics")
        metrics = self.evaluate(X, y)
        
        # Get feature importance
        feature_importance = self.get_feature_importance()
        
        # Create metadata
        processing_time = time.time() - start_time
        metadata = self._create_ml_metadata(
            X.shape, y.shape[0], processing_time,
            cv_score=cv_results.mean_metrics.get('r2') if cv_results else None,
            cv_std=cv_results.std_metrics.get('r2') if cv_results else None
        )
        
        # Create result
        result = MLResult(
            labels=predictions,  # Using labels field for predictions
            metadata=metadata,
            statistics=metrics,
            predictions=predictions,
            model_metadata=self.model_metadata,
            cv_results=cv_results,
            feature_importance=feature_importance
        )
        
        self._update_progress(100, 100, "ML analysis complete")
        
        return result
    
    def _prepare_ml_data(self, 
                        data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                        target_column: str,
                        feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for ML analysis.
        
        Args:
            data: Input data
            target_column: Target column name
            feature_columns: Feature column names
            
        Returns:
            Tuple of (features, target, feature_names)
        """
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # Extract target
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        y = data[target_column]
        
        # Extract features
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        X = data[feature_columns]
        
        return X, y, feature_columns
    
    def validate_ml_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[bool, List[str]]:
        """
        Validate ML input data.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []
        
        # Check shapes
        if len(X) != len(y):
            issues.append(f"Feature and target lengths don't match: {len(X)} vs {len(y)}")
        
        # Check for empty data
        if len(X) == 0:
            issues.append("No samples in data")
        
        if X.shape[1] == 0:
            issues.append("No features in data")
        
        # Check for all NaN columns
        all_nan_features = X.columns[X.isna().all()].tolist()
        if all_nan_features:
            issues.append(f"Features with all NaN values: {all_nan_features}")
        
        # Check target
        if y.isna().all():
            issues.append("Target has all NaN values")
        
        # Check for constant features
        constant_features = X.columns[X.nunique() == 1].tolist()
        if constant_features:
            issues.append(f"Constant features detected: {constant_features}")
        
        return len(issues) == 0, issues
    
    def _create_ml_metadata(self, 
                           feature_shape: Tuple[int, int],
                           n_samples: int,
                           processing_time: float,
                           **kwargs) -> ModelMetadata:
        """Create ML-specific metadata."""
        self.model_metadata = ModelMetadata(
            model_type=self.model_type.value,
            algorithm=self.__class__.__name__,
            training_samples=n_samples,
            training_features=feature_shape[1],
            feature_names=self.feature_names,
            target_name=self.target_name,
            training_time=processing_time,
            training_date=datetime.now(),
            cv_score=kwargs.get('cv_score'),
            cv_std=kwargs.get('cv_std'),
            hyperparameters=self.get_model_params(),
            training_metrics=kwargs.get('training_metrics', {}),
            feature_importance=self.get_feature_importance()
        )
        
        # Also create standard metadata for compatibility
        analysis_metadata = AnalysisMetadata(
            analysis_type=f"ML_{self.model_type.value.upper()}",
            input_shape=feature_shape,
            parameters=self.get_model_params(),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        # Store both metadata
        self.model_metadata.analysis_metadata = analysis_metadata
        
        return self.model_metadata
    
    def save_model(self, path: Path) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_state = {
            'model': self.model,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_metadata': self.model_metadata,
            'model_params': self.get_model_params(),
            'model_type': self.model_type.value,
            'analyzer_class': self.__class__.__name__
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        
        logger.info(f"Model saved to {path}")
        
        # Also save metadata as JSON
        metadata_path = path.with_suffix('.json')
        metadata_dict = {
            'model_type': self.model_type.value,
            'algorithm': self.__class__.__name__,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'training_samples': self.model_metadata.training_samples if self.model_metadata else None,
            'training_date': self.model_metadata.training_date.isoformat() if self.model_metadata else None,
            'model_params': self.get_model_params()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def load_model(self, path: Path) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load the model from
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        # Restore model state
        self.model = model_state['model']
        self.feature_names = model_state['feature_names']
        self.target_name = model_state['target_name']
        self.model_metadata = model_state.get('model_metadata')
        self.model_type = ModelType(model_state.get('model_type', 'regression'))
        self.is_fitted = True
        
        # Set model parameters if available
        if 'model_params' in model_state:
            self.set_model_params(**model_state['model_params'])
        
        logger.info(f"Model loaded from {path}")
    
    def cross_validate(self,
                      X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, pd.Series],
                      cv_strategy: Any,
                      metrics: Optional[List[str]] = None,
                      **kwargs) -> CVResult:
        """
        Perform cross-validation with spatial awareness.
        
        Args:
            X: Feature matrix
            y: Target values
            cv_strategy: Cross-validation strategy
            metrics: Metrics to compute
            **kwargs: Additional parameters
            
        Returns:
            CVResult with detailed fold information
        """
        if metrics is None:
            metrics = ['r2', 'rmse', 'mae'] if self.model_type == ModelType.REGRESSION else ['accuracy', 'f1']
        
        folds = []
        all_metrics = {metric: [] for metric in metrics}
        
        # Get CV splits
        n_splits = cv_strategy.get_n_splits()
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_strategy.split(X, y)):
            fold_start = time.time()
            
            # Get fold data
            X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
            X_test = X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx]
            y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            y_test = y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]
            
            # Train model
            self.fit(X_train, y_train, **kwargs)
            
            # Evaluate
            fold_metrics = self.evaluate(X_test, y_test, metrics)
            
            # Store metrics
            for metric in metrics:
                if metric in fold_metrics:
                    all_metrics[metric].append(fold_metrics[metric])
            
            # Create fold info
            fold = CVFold(
                fold_id=fold_idx,
                train_indices=train_idx,
                test_indices=test_idx,
                train_size=len(train_idx),
                test_size=len(test_idx),
                metrics=fold_metrics,
                training_time=time.time() - fold_start
            )
            folds.append(fold)
            
            self._update_progress(
                20 + int(40 * (fold_idx + 1) / n_splits), 
                100, 
                f"Completed CV fold {fold_idx + 1}/{n_splits}"
            )
        
        # Calculate aggregate metrics
        mean_metrics = {metric: np.mean(values) for metric, values in all_metrics.items() if values}
        std_metrics = {metric: np.std(values) for metric, values in all_metrics.items() if values}
        
        # Create CV result
        cv_result = CVResult(
            strategy=cv_strategy.__class__.__name__,
            n_folds=n_splits,
            folds=folds,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            total_time=sum(fold.training_time for fold in folds)
        )
        
        return cv_result
    
    @abstractmethod
    def fit(self, 
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            sample_weight: Optional[np.ndarray] = None,
            **kwargs) -> 'BaseMLAnalyzer':
        """Fit the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        """Make predictions. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def evaluate(self,
                 X: Union[np.ndarray, pd.DataFrame],
                 y: Union[np.ndarray, pd.Series],
                 metrics: Optional[List[str]] = None,
                 **kwargs) -> Dict[str, float]:
        """Evaluate model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def set_model_params(self, **params) -> 'BaseMLAnalyzer':
        """Set model parameters. Must be implemented by subclasses."""
        pass
    
    def fit_predict(self,
                    X: Union[np.ndarray, pd.DataFrame],
                    y: Union[np.ndarray, pd.Series],
                    sample_weight: Optional[np.ndarray] = None,
                    **kwargs) -> np.ndarray:
        """
        Fit model and return predictions on training data.
        
        Default implementation, can be overridden for efficiency.
        """
        self.fit(X, y, sample_weight, **kwargs)
        return self.predict(X, **kwargs)
    
    def explain_predictions(self,
                           X: Union[np.ndarray, pd.DataFrame],
                           method: str = "shap",
                           **kwargs) -> Dict[str, Any]:
        """
        Explain predictions using specified method.
        
        Default implementation returns feature importance.
        Subclasses can override for more sophisticated explanations.
        """
        if method == "feature_importance":
            return {
                'method': 'feature_importance',
                'values': self.get_feature_importance()
            }
        else:
            raise NotImplementedError(f"Explanation method '{method}' not implemented")
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default ML parameters."""
        return {
            'model_type': self.model_type.value,
            'save_predictions': self.save_predictions,
            'save_model': self.save_model
        }